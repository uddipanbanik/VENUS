#pragma once

// Construct the global and MPI-local state before a spherical kinetic run.
// Inputs are validated model/grid controls, optional binary initial data or a full
// restart, and the MPI decomposition. Build coordinate measures first, then load
// the distribution and any passive tracer on those measures. Restore evolved
// clocks, carriers and modal state from a compatible restart. Output setup keeps
// the source checkpoint separate and validates CSV schemas before appending.

#include "ev/solver_state.hpp"
#include "ev/operator_diagnostics.hpp"
#include "ev/vlasov_poisson.hpp"

namespace ev {

/// Verifies that a tracer IC lives on exactly the same phase-space grid as the total IC.
void validate_matching_initial_data_grids(const InitialDataRWL& total, const InitialDataRWL& tracer) {
    if (total.Nr != tracer.Nr || total.Nw != tracer.Nw || total.NL != tracer.NL
        || total.r != tracer.r || total.w != tracer.w || total.L != tracer.L) {
        throw std::runtime_error("tracer_initial_data_file grid does not match initial_data_file.");
    }
}

std::filesystem::path absolute_normal_path(const std::filesystem::path& path) {
    return std::filesystem::absolute(path).lexically_normal();
}

/// Compares two paths after making them absolute and lexically normalized.
bool same_filesystem_path(const std::filesystem::path& a, const std::filesystem::path& b) {
    return absolute_normal_path(a) == absolute_normal_path(b);
}

/// Reports whether a directory exists and contains any entries.
bool directory_has_entries(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return false;
    }
    if (!std::filesystem::is_directory(path)) {
        throw std::runtime_error("Expected a directory but found a file: " + path.string());
    }
    return std::filesystem::directory_iterator(path) != std::filesystem::directory_iterator();
}

/// Identifies files that are allowed to appear in a solver restart directory.
bool is_restart_artifact_name(const std::filesystem::path& path) {
    const std::string name = path.filename().string();
    return name == "restart_state.bin"
        || name.rfind("restart_state.bin.rank", 0) == 0
        || name.rfind("restart_backup_", 0) == 0;
}

/// Refuses a restart directory containing unrelated files that could be overwritten.
void validate_restart_directory_contents(const std::filesystem::path& restart_dir) {
    if (!std::filesystem::exists(restart_dir)) {
        return;
    }
    if (!std::filesystem::is_directory(restart_dir)) {
        throw std::runtime_error("restart_dir exists but is not a directory: " + restart_dir.string());
    }
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(restart_dir)) {
        if (!entry.is_regular_file() || !is_restart_artifact_name(entry.path())) {
            throw std::runtime_error(
                "Refusing restart_dir with non-restart artifact: " + entry.path().string());
        }
    }
}

/// Fresh runs may write into a new output directory, or one containing only the restart subdirectory.
void validate_output_directory_is_new(const SolverState& state) {
    if (!std::filesystem::exists(state.outdir)) {
        return;
    }
    if (!std::filesystem::is_directory(state.outdir)) {
        throw std::runtime_error("output_dir exists but is not a directory: " + state.outdir.string());
    }
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(state.outdir)) {
        if (same_filesystem_path(entry.path(), state.restart_dir)) {
            continue;
        }
        throw std::runtime_error(
            "Refusing to start in an existing output_dir. Choose a new output_dir or an explicit restart continuation: "
            + state.outdir.string());
    }
}

/// Restart continuations may append CSVs, but restarts must be staged in a separate destination.
void validate_output_directory_for_restart_append(const SolverState& state) {
    if (!std::filesystem::exists(state.outdir)) {
        return;
    }
    if (!std::filesystem::is_directory(state.outdir)) {
        throw std::runtime_error("output_dir exists but is not a directory: " + state.outdir.string());
    }
    if (same_filesystem_path(state.outdir, state.restart_dir)) {
        throw std::runtime_error("restart_dir must be separate from output_dir: " + state.restart_dir.string());
    }
}

/// Applies all output/restart overwrite guards before any files are created.
// A continuation reads its source checkpoint while writing a distinct restart
// destination. Validate all destinations before creating the run output streams.
void protect_run_directories(const SolverState& state) {
    if (same_filesystem_path(state.outdir, state.restart_dir)) {
        throw std::runtime_error("restart_dir must be separate from output_dir: " + state.restart_dir.string());
    }
    if (same_filesystem_path(state.restart_tmp_dir, state.restart_dir)) {
        throw std::runtime_error("Internal error: restart staging directory equals restart_dir.");
    }
    validate_restart_directory_contents(state.restart_dir);
    if (state.global_p.restart_from.empty()) {
        validate_output_directory_is_new(state);
        if (directory_has_entries(state.restart_dir)) {
            throw std::runtime_error(
                "Refusing fresh run with an existing restart_dir. Choose a new restart_dir: "
                + state.restart_dir.string());
        }
    } else {
        validate_output_directory_for_restart_append(state);
        const std::filesystem::path source_restart_dir =
            absolute_normal_path(std::filesystem::path(state.global_p.restart_from).parent_path());
        if (same_filesystem_path(source_restart_dir, state.restart_dir)) {
            throw std::runtime_error(
                "Refusing to write a restarted run into the source restart directory. "
                "Choose a new restart_dir so the source checkpoint remains untouched: "
                + state.restart_dir.string());
        }
        if (directory_has_entries(state.restart_dir)) {
            throw std::runtime_error(
                "Refusing restarted run with a non-empty destination restart_dir. Choose a new restart_dir: "
                + state.restart_dir.string());
        }
    }
}

/// Builds grids, quadrature weights, decomposition metadata, and output paths from parameters.
SolverState create_solver_state(Params global_p, const MpiContext& mpi) {
    SolverState state;
    state.global_p = global_p;
    state.mpi = mpi;
    state.binary_ic = using_binary_initial_data(state.global_p);

    if (state.binary_ic) {
        state.custom_ic = read_initial_data_rwl_metadata(state.global_p.initial_data_file);
        if (state.global_p.tracer_initial_data) {
            state.tracer_ic = read_initial_data_rwl_metadata(state.global_p.tracer_initial_data_file);
            validate_matching_initial_data_grids(state.custom_ic, state.tracer_ic);
            state.has_tracer = true;
        }
        state.global_p.Nr = state.custom_ic.Nr;
        state.global_p.Nw = state.custom_ic.Nw;
        state.global_p.NL = state.custom_ic.NL;
        state.global_p.r_min = state.custom_ic.r.front();
        state.global_p.r_max = state.custom_ic.r.back();
        state.global_p.L_max = state.custom_ic.L.back();
        state.global_p.w_max = std::max(std::abs(state.custom_ic.w.front()), std::abs(state.custom_ic.w.back()));
        state.global_p.radial_grid = "custom";
    }
    validate_params(state.global_p);
    validate_mpi_execution_mode(state.global_p, state.mpi);

    // Each rank owns a contiguous angular-momentum or eta slab. Radial/velocity
    // coordinates are shared, and field moments later sum the angular contributions.
    state.decomp = make_l_decomposition(state.global_p.NL, state.mpi);
    state.local_p = state.global_p;
    state.local_p.NL = state.decomp.count;
    if (using_static_hamiltonian_potential_macrostep(state.local_p)
        && state.local_p.vp_sldg_initial_transport_dt > 0.0) {
        state.vp_sldg_transport_dt_cap = std::min(
            state.local_p.dt_max,
            state.local_p.vp_sldg_initial_transport_dt);
    }

    PhaseSpaceAxis cell_average_radial_axis;
    // Conservative cell averages use arithmetic coordinate centroids and widths.
    // The nodal branch instead obtains its interpolation nodes from the initial data.
    if (using_cell_average_eta_grid(state.global_p)) {
        cell_average_radial_axis = make_positive_radial_cell_axis(state.global_p);
        state.r = cell_average_radial_axis.center;
    } else {
        state.r = state.binary_ic ? state.custom_ic.r : make_radial_grid(state.global_p);
    }
    state.x_r = radial_coordinate_grid(state.r, state.global_p);
    if (using_cell_average_eta_grid(state.global_p)) {
        const PhaseSpaceCellGeometry global_geometry =
            make_cell_average_eta_geometry(cell_average_radial_axis, state.global_p);
        state.has_cell_average_geometry = true;
        state.kinetic_geometry.radial = global_geometry.radial;
        state.kinetic_geometry.radial_velocity = global_geometry.radial_velocity;
        state.kinetic_geometry.eta = slice_phase_space_axis(
            global_geometry.eta, state.decomp.offset, state.decomp.count);
        state.kinetic_geometry.L = slice_vector(global_geometry.L, state.decomp);
        state.kinetic_geometry.legacy_Lq =
            slice_vector(global_geometry.legacy_Lq, state.decomp);
        state.w = state.kinetic_geometry.radial_velocity.center;
        state.L = state.kinetic_geometry.L;
        state.Lq = state.kinetic_geometry.legacy_Lq;
        state.rq = state.kinetic_geometry.radial.width;
        state.wq = state.kinetic_geometry.radial_velocity.width;
    } else {
        state.w = state.binary_ic
            ? state.custom_ic.w
            : stretched_w_grid(
                state.global_p.w_max,
                state.global_p.w_stretch,
                state.global_p.w_center,
                state.global_p.Nw);
        const std::vector<double> L_global = state.binary_ic
            ? state.custom_ic.L
            : linspace(0.0, state.global_p.L_max, state.global_p.NL);
        const std::vector<double> Lq_global = quadrature_weights(L_global);
        state.L = slice_vector(L_global, state.decomp);
        state.Lq = slice_vector(Lq_global, state.decomp);
        state.rq = using_conservative_phase_space_cells(state.global_p)
            ? radial_control_volume_widths(state.r, state.global_p)
            : quadrature_weights(state.r);
        state.wq = quadrature_weights(state.w);
    }
    state.radial_step = min_spacing(state.x_r);
    state.min_dw = min_spacing(state.w);
    state.outdir = absolute_normal_path(state.global_p.output_dir);
    state.restart_dir = state.global_p.restart_dir.empty()
        ? (state.outdir / "restarts")
        : absolute_normal_path(state.global_p.restart_dir);
    state.restart_tmp_dir = state.outdir / ".restart_staging";
    state.restart_path = state.restart_dir / "restart_state.bin";
    state.global_p.output_dir = state.outdir.string();
    state.global_p.restart_dir = state.restart_dir.string();
    return state;
}

/// Initializes f either from a restart, a binary IC file, or analytic initial data.
// A full restart has priority over fresh binary or analytic initial data.
// Its saved time, step, tracer, unresolved carriers and modal state travel together.
void initialize_distribution(SolverState& state) {
    if (!state.global_p.restart_from.empty()) {
        const bool requested_tracer = state.has_tracer || state.global_p.tracer_initial_data;
        bool restart_has_tracer = false;
        state.f = read_restart_state_distributed(
            state.global_p.restart_from,
            state.global_p,
            state.local_p,
            state.decomp,
            state.mpi,
            state.t,
            state.step,
            &state.tracer_f,
            &restart_has_tracer,
            using_delayed_inner_return(state.local_p)
                ? &state.inner_carrier
                : nullptr,
            using_delayed_inner_return(state.local_p)
                ? &state.tracer_inner_carrier
                : nullptr,
            &state.hamiltonian_sldg_modes,
            &state.hamiltonian_sldg_modes_valid,
            &state.tracer_hamiltonian_sldg_modes,
            &state.tracer_hamiltonian_sldg_modes_valid);
        if (!using_hamiltonian_sldg_transport(state.local_p)) {
            state.hamiltonian_sldg_modes.clear();
            state.tracer_hamiltonian_sldg_modes.clear();
            state.hamiltonian_sldg_modes_valid = false;
            state.tracer_hamiltonian_sldg_modes_valid = false;
        }
        if (requested_tracer && !restart_has_tracer) {
            throw std::runtime_error("Restart file does not contain tracer state.");
        }
        if (restart_has_tracer) {
            if (state.global_p.adaptive_w) {
                throw std::runtime_error("Tracer restart currently requires adaptive_w=false.");
            }
            state.has_tracer = true;
            state.global_p.tracer_initial_data = true;
        } else {
            state.has_tracer = false;
            state.global_p.tracer_initial_data = false;
        }
        if (is_root(state.mpi)) {
            std::cout << "Restarted from " << state.global_p.restart_from
                      << " at step=" << state.step << " t=" << std::setprecision(17) << state.t
                      << " tracer=" << (state.has_tracer ? "yes" : "no") << "\n";
        }
        return;
    }

    if (state.binary_ic) {
        state.custom_ic.f = read_initial_data_rwl_local_values(
            state.global_p.initial_data_file,
            state.custom_ic,
            state.decomp);
        state.f = custom_initial_distribution_rwl(state.custom_ic, state.local_p);
        if (state.has_tracer) {
            state.tracer_ic.f = read_initial_data_rwl_local_values(
                state.global_p.tracer_initial_data_file,
                state.tracer_ic,
                state.decomp);
            state.tracer_f = custom_initial_distribution_rwl(state.tracer_ic, state.local_p);
        }
        // Requested mass normalization rescales total and tracer distributions by the
        // same factor. The measured mass uses the active model and its field equations.
        if (state.global_p.custom_ic_normalize_mass) {
            const Moments moments = using_vlasov_poisson(state.global_p)
                ? compute_global_vp_moments(
                    state.f, state.r, state.w, state.L, state.wq, state.Lq, state.local_p, state.mpi)
                : compute_global_moments(
                    state.f, state.r, state.w, state.L, state.wq, state.Lq, state.local_p, state.mpi);
            const double matter_mass = using_vlasov_poisson(state.global_p)
                ? vp_matter_mass(moments, state.r, state.global_p)
                : (solve_metric(moments, state.r, state.global_p).m.back() - state.global_p.central_mass);
            if (!(matter_mass > 0.0) || !std::isfinite(matter_mass)) {
                throw std::runtime_error("Cannot normalize custom IC: computed matter mass is not positive.");
            }
            const double scale = state.global_p.custom_ic_target_mass / matter_mass;
            for (double& value : state.f) {
                value *= scale;
            }
            for (double& value : state.tracer_f) {
                value *= scale;
            }
            if (is_root(state.mpi)) {
                std::cout << "Normalized custom initial data mass from "
                          << std::setprecision(12) << matter_mass
                          << " to " << state.global_p.custom_ic_target_mass << "\n";
            }
        }
        return;
    }

    std::vector<double> mu_seed(state.global_p.Nr, 0.0);
    state.f = using_cell_average_eta_grid(state.global_p)
        ? initial_distribution_cell_average_eta(
            state.kinetic_geometry, mu_seed, state.x_r, state.local_p)
        : initial_distribution(
            state.r,
            state.w,
            state.L,
            state.wq,
            state.Lq,
            mu_seed,
            state.x_r,
            state.local_p);
    if (using_vlasov_poisson(state.global_p)) {
        return;
    }
    // Metric-dependent analytic relativistic data are iterated with the metric
    // sourced by their own moments. Newtonian initial data have already returned.
    for (int iter = 0; iter < state.global_p.initial_metric_iters; ++iter) {
        const Moments moments = compute_global_moments(
            state.f, state.r, state.w, state.L, state.wq, state.Lq, state.local_p, state.mpi);
        const Metric metric = solve_metric(moments, state.r, state.global_p);
        state.f = initial_distribution(
            state.r, state.w, state.L, state.wq, state.Lq, metric.mu, state.x_r, state.local_p);
    }
}

/// Creates the output directory and writes metadata, initial phase projection, and initial restart.
// Publish metadata and the initial restart only after state initialization.
// MPI barriers make the root-created destination available to every writing rank.
void write_initial_outputs(const SolverState& state) {
    if (is_root(state.mpi)) {
        std::filesystem::create_directories(state.outdir);
        std::filesystem::create_directories(state.restart_dir);
        std::filesystem::create_directories(state.restart_tmp_dir);
    }
    mpi_barrier(state.mpi);
    if (is_root(state.mpi)) {
        write_metadata(state.outdir / "metadata.json", state.global_p, state.local_p, state.decomp, state.mpi);
    }
    if (!state.global_p.restart_from.empty()) {
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
        return;
    }
    {
        std::vector<double> g = phase_projection_global(state.f, state.L, state.Lq, state.local_p, state.mpi);
        if (is_root(state.mpi)) {
            write_phase_space(state.outdir / "phase_space_initial.csv", state.r, state.w, g, state.global_p);
        }
    }
    if (state.has_tracer) {
        std::vector<double> g = phase_projection_global(state.tracer_f, state.L, state.Lq, state.local_p, state.mpi);
        if (is_root(state.mpi)) {
            write_phase_space(state.outdir / "tracer_phase_space_initial.csv", state.r, state.w, g, state.global_p);
        }
    }
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

/// Opens profile, time-series, adaptive-w, and optional phase-snapshot CSV streams on the root rank.
SolverOutputStreams open_output_streams(const SolverState& state) {
    SolverOutputStreams streams;
    if (!is_root(state.mpi)) {
        return streams;
    }

    const bool append = !state.global_p.restart_from.empty();
    // A continuing CSV retains exactly one compatible header. Checking the schema
    // protects the meaning and order of all subsequently appended diagnostics.
    auto open_csv = [&](std::ofstream& stream, const std::filesystem::path& path, const char* header) {
        const bool has_existing_rows =
            append && std::filesystem::exists(path) && std::filesystem::file_size(path) > 0;
        if (has_existing_rows) {
            std::ifstream existing(path);
            std::string existing_header;
            std::getline(existing, existing_header);
            std::string expected_header(header);
            if (!expected_header.empty() && expected_header.back() == '\n') {
                expected_header.pop_back();
            }
            if (!existing || existing_header != expected_header) {
                throw std::runtime_error(
                    "Existing CSV schema does not match this executable: "
                    + path.string()
                    + ". Continue into a new output_dir and retain the prior segment.");
            }
        }
        stream.open(path, append ? std::ios::app : std::ios::out);
        if (!stream) {
            throw std::runtime_error("Could not open output CSV: " + path.string());
        }
        if (!has_existing_rows) {
            stream << header;
            if (!stream) {
                throw std::runtime_error("Could not write output CSV header: " + path.string());
            }
            flush_output_checked(stream, path.filename().string());
        }
    };

    open_csv(
        streams.profiles,
        state.outdir / "profiles.csv",
        "time,step,r,rho,p,pt,j,m,mu,lambda,gap\n");

    if (state.has_tracer) {
        open_csv(
            streams.tracer_profiles,
            state.outdir / "tracer_profiles.csv",
            "time,step,r,rho,p,pt,j\n");
    }

    open_csv(
        streams.history,
        state.outdir / "history.csv",
        "step,time,dt,matter_mass,max_rho,min_gap,"
        "last_outer_escape_est,last_w_low_escape_est,last_w_high_escape_est,"
        "cum_outer_escape_est,cum_w_low_escape_est,cum_w_high_escape_est,"
        "last_outer_cells,last_w_low_cells,last_w_high_cells,"
        "w_max,min_dw,hidden_carrier_mass,hidden_carrier_mean_density,"
        "hidden_carrier_packets,"
        "op_vlasov_rms,op_collision_rms,op_collision_over_vlasov,"
        "op_vlasov_rms_r_lt_1,op_collision_rms_r_lt_1,op_collision_over_vlasov_r_lt_1,"
        "op_vlasov_rms_r_1_3,op_collision_rms_r_1_3,op_collision_over_vlasov_r_1_3,"
        "op_vlasov_rms_r_ge_3,op_collision_rms_r_ge_3,op_collision_over_vlasov_r_ge_3,"
        "vp_orbit_lc_retries_cumulative,vp_orbit_lc_accepted_cumulative,"
        "vp_orbit_lc_unresolved_cumulative,"
        "vp_orbit_wall_retries_cumulative,vp_orbit_wall_accepted_cumulative,"
        "vp_orbit_wall_unresolved_cumulative,"
        "vp_orbit_maximum_ordinary_energy_error_cumulative,"
        "vp_orbit_maximum_final_energy_error_cumulative,"
        "vp_potential_variation_timescale,"
        "vp_potential_variation_dt_limit\n");
    streams.history << std::setprecision(17);

    // This ledger separates transport, collision, remap and hidden-carrier mass
    // changes so a complete source/boundary budget can be reconstructed.
    open_csv(
        streams.operator_mass_budget,
        state.outdir / "operator_mass_budget.csv",
        "step,time,operator,operator_dt,mass_before,mass_after,mass_delta,relative_mass_delta,"
        "radial_momentum_before,radial_momentum_after,radial_momentum_delta,"
        "kinetic_energy_before,kinetic_energy_after,kinetic_energy_delta,"
        "relative_kinetic_energy_delta,"
        "remap_attempted_event_density,remap_skipped_event_density,"
        "remap_skipped_fraction,remap_skipped_w_event_density,"
        "remap_skipped_L_event_density,max_shell_remap_skipped_fraction,"
        "remap_skipped_domain_event_density,remap_skipped_projection_event_density,"
        "remap_skipped_inactive_event_density,"
        "maximum_projection_error,"
        "sampled_dense_reactions,sampled_complement_reactions,"
        "sampled_complement_draws,sampled_retained_collision_draws,"
        "sampled_ntc_fallback_shells,sampled_min_samples_per_batch,"
        "sampled_max_samples_per_batch,sampled_max_adaptive_rounds,"
        "sampled_unconverged_shells,sampled_inactive_shells,"
        "sampled_max_rhs_relative_error,"
        "sampled_max_rhs_gross_error,"
        "sampled_max_heat_relative_error,sampled_max_heat_gross_error,"
        "sampled_max_energy_invariant_error,"
        "minimum_mean_free_path,"
        "maximum_radial_cell_width_over_mean_free_path,"
        "nonlocal_collision_cells,"
        "reversible_reaction_updates,"
        "reversible_explicit_fast_path_fallbacks,"
        "reversible_implicit_reaction_updates,"
        "reversible_implicit_iterations,"
        "sldg_source_mass,sldg_deposited_mass,sldg_escaped_mass,"
        "sldg_carrier_mass_before,sldg_carrier_mass_after,"
        "sldg_carrier_captured_mass,sldg_carrier_released_mass,"
        "sldg_carrier_release_deposited_mass,sldg_carrier_mass_residual,"
        "sldg_carrier_packets_before,sldg_carrier_packets_after,"
        "sldg_carrier_packets_captured,sldg_carrier_packets_released,"
        "sldg_carrier_compressions,"
        "sldg_carrier_compression_packets_before,"
        "sldg_carrier_compression_packets_after,"
        "sldg_carrier_compression_mass_delta,"
        "sldg_carrier_compression_momentum_delta,"
        "sldg_carrier_compression_energy_delta,"
        "sldg_carrier_compression_return_time_moment_delta,"
        "sldg_quadrature_fallback_mass,sldg_quadrature_fallback_fraction,"
        "sldg_maximum_area_defect,sldg_maximum_map_defect,"
        "sldg_maximum_rejected_area_defect,sldg_maximum_rejected_map_defect,"
        "sldg_transported_subcells,sldg_traced_rectangles,"
        "sldg_rejected_rectangles,sldg_fallback_subcells,"
        "sldg_maximum_source_work,sldg_largest_fallback_mass,"
        "sldg_largest_fallback_r,sldg_largest_fallback_w,"
        "sldg_largest_fallback_L,sldg_largest_fallback_refinement,"
        "sldg_largest_fallback_work,"
        "sldg_characteristic_trace_requests,sldg_characteristic_traces,"
        "sldg_reused_characteristic_points,"
        "sldg_midpoint_predictor_mass_before,"
        "sldg_midpoint_predictor_mass_after,"
        "sldg_midpoint_predictor_escaped_mass,"
        "sldg_midpoint_predictor_hidden_mass_before,"
        "sldg_midpoint_predictor_hidden_mass_after,"
        "sldg_midpoint_predictor_hidden_mass_change,"
        "sldg_midpoint_predictor_limited_faces,"
        "sldg_hamiltonian_reconstructed_cells,"
        "sldg_hamiltonian_limited_cells,"
        "sldg_hamiltonian_projected_limited_cells,"
        "sldg_hamiltonian_rank_deficient_cells,"
        "sldg_hamiltonian_phase_degenerate_cells,"
        "sldg_hamiltonian_eta_limited_cells,"
        "sldg_hamiltonian_maximum_phase_gradient_fraction,"
        "sldg_hamiltonian_reference_mass_fraction,"
        "sldg_hamiltonian_residual_l1_fraction,"
        "sldg_hamiltonian_reference_active_planes,"
        "sldg_hamiltonian_minimum_positive_reference_scale,"
        "sldg_hamiltonian_positive_reference_scaled_planes,"
        "sldg_hamiltonian_positivity_redistributed_mass,"
        "sldg_hamiltonian_positivity_redistributed_fraction\n");
    streams.operator_mass_budget << std::setprecision(17);

    open_csv(
        streams.w_grid_history,
        state.outdir / "w_grid_history.csv",
        "step,time,old_w_max,new_w_max,active_w_min,active_w_max,min_dw,"
        "mass_before,mass_after,rel_mass_change\n");
    streams.w_grid_history << std::setprecision(17);

    if (state.global_p.phase_every > 0) {
        open_csv(
            streams.phase_snapshots,
            state.outdir / "phase_space_snapshots.csv",
            "time,step,r_index,w_index,r,w,g\n");
        if (state.has_tracer) {
            open_csv(
                streams.tracer_phase_snapshots,
                state.outdir / "tracer_phase_space_snapshots.csv",
                "time,step,r_index,w_index,r,w,g\n");
        }
    }
    return streams;
}

}  // namespace ev
