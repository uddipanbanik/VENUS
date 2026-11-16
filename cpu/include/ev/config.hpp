#pragma once

// Read the CPU solver physical and numerical parameter record.
// Defaults are filled first, selected configuration files next, environment
// overrides next and explicit command-line values last. The returned Params
// then enters model-aware setup validation. Input paths in a configuration
// file are resolved relative to that file; explicit command-line paths retain
// the caller interpretation. This layer parses controls and launch layout;
// physical evolution and state reconstruction are performed by other modules.

#include "ev/parallel.hpp"
#include "ev/mpi.hpp"

namespace ev {

/// Returns a copy of a string with leading and trailing whitespace removed.
std::string trim_copy(const std::string& s) {
    std::size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) {
        ++first;
    }
    std::size_t last = s.size();
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1]))) {
        --last;
    }
    return s.substr(first, last - first);
}

/// Removes unquoted # and ; comments from one config-file line.
// Recognize comment delimiters only outside quoted text, so a path
// or literal parameter value can contain # or semicolon characters.
std::string strip_inline_comment(const std::string& s) {
    bool in_single = false;
    bool in_double = false;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '\'' && !in_double) {
            in_single = !in_single;
        } else if (c == '"' && !in_single) {
            in_double = !in_double;
        } else if (!in_single && !in_double && (c == '#' || c == ';')) {
            return s.substr(0, i);
        }
    }
    return s;
}

/// Removes one surrounding pair of single or double quotes after trimming whitespace.
std::string unquote_copy(const std::string& s) {
    std::string out = trim_copy(s);
    if (out.size() >= 2
        && ((out.front() == '"' && out.back() == '"')
            || (out.front() == '\'' && out.back() == '\''))) {
        out = out.substr(1, out.size() - 2);
    }
    return out;
}

/// Normalizes CLI/config keys by removing leading dashes and replacing underscores with hyphens.
// Config spelling is case-insensitive after this normalization;
// underscores and hyphens name the same configured key. String values
// retain their own physical/model spelling.
std::string normalize_key(std::string key) {
    key = trim_copy(key);
    while (key.rfind("--", 0) == 0) {
        key.erase(0, 2);
    }
    for (char& c : key) {
        if (c == '_') {
            c = '-';
        } else {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return key;
}

/// Parses common textual booleans and reports the offending key on failure.
bool parse_bool_value(const std::string& text, const std::string& key) {
    std::string value = normalize_key(text);
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    throw std::runtime_error("Expected boolean value for " + key + ", got '" + text + "'.");
}

/// Returns a nonempty environment variable value or nullptr when it is unset/empty.
const char* getenv_nonempty(const char* name) {
    const char* value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? value : nullptr;
}

/// Parses a strictly positive integer from text used in environment overrides.
int parse_positive_int_text(const std::string& text, const std::string& source) {
    const std::string value = trim_copy(text);
    std::size_t pos = 0;
    int out = 0;
    try {
        out = std::stoi(value, &pos);
    } catch (const std::exception&) {
        throw std::runtime_error("Expected a positive integer for " + source + ", got '" + text + "'.");
    }
    if (pos != value.size() || out <= 0) {
        throw std::runtime_error("Expected a positive integer for " + source + ", got '" + text + "'.");
    }
    return out;
}

/// Reads a positive integer environment variable, returning zero when absent.
int env_positive_int_or_zero(const char* name) {
    const char* value = getenv_nonempty(name);
    if (value == nullptr) {
        return 0;
    }
    return parse_positive_int_text(value, std::string("environment variable ") + name);
}

/// Returns the first positive integer among a priority list of environment variables.
int first_positive_env_int(const std::vector<std::string>& names) {
    for (const std::string& name : names) {
        const int value = env_positive_int_or_zero(name.c_str());
        if (value > 0) {
            return value;
        }
    }
    return 0;
}

/// Applies solver environment overrides for output, restart, threads, and launch safety.
// Environment values deliberately override loaded configuration values
// for output, restarts and worker count. The explicit command-line pass
// below still has the final precedence.
void apply_environment_overrides(Params& p) {
    if (const char* output_dir = getenv_nonempty("EV_OUTPUT_DIR")) {
        p.output_dir = output_dir;
    }
    if (const char* output_dir = getenv_nonempty("EV_SOLVER_OUTPUT_DIR")) {
        p.output_dir = output_dir;
    }
    if (const char* restart_dir = getenv_nonempty("EV_RESTART_DIR")) {
        p.restart_dir = restart_dir;
    }
    if (const char* restart_from = getenv_nonempty("EV_RESTART_FROM")) {
        p.restart_from = restart_from;
    }
    const int forced_threads = first_positive_env_int({"EV_THREADS", "EV_SOLVER_THREADS"});
    if (forced_threads > 0) {
        p.threads = forced_threads;
    }
    if (const char* allow = getenv_nonempty("EV_ALLOW_MULTI_RANK_LAUNCH")) {
        p.allow_multi_rank_launch = parse_bool_value(allow, "EV_ALLOW_MULTI_RANK_LAUNCH");
    }
}

/// True when either environment alias for the solver output directory is set.
bool environment_overrides_output_dir() {
    return getenv_nonempty("EV_OUTPUT_DIR") != nullptr || getenv_nonempty("EV_SOLVER_OUTPUT_DIR") != nullptr;
}

/// True when the restart destination is explicitly set by the environment.
bool environment_overrides_restart_dir() {
    return getenv_nonempty("EV_RESTART_DIR") != nullptr;
}

/// Chooses a concrete per-rank thread count and configures OpenMP when enabled.
// Resolve an automatic worker request from the allocation/environment
// or local backend. Disabling dynamic OpenMP teams keeps the selected
// per-rank worker count explicit.
void finalize_parallel_runtime(Params& p) {
    if (p.threads <= 0) {
        const int env_threads = first_positive_env_int({"OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK"});
        p.threads = (env_threads > 0) ? env_threads : max_backend_threads();
    }
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(std::max(1, p.threads));
#endif
}

/// Detects Slurm/MPI world size from environment variables in non-MPI builds.
int detected_multi_rank_size() {
    int world_size = 1;
    for (const char* name : {
             "SLURM_NTASKS",
             "OMPI_COMM_WORLD_SIZE",
             "PMI_SIZE",
             "PMIX_SIZE",
             "MPI_LOCALNRANKS"}) {
        const int value = env_positive_int_or_zero(name);
        world_size = std::max(world_size, value);
    }
    return world_size;
}

/// Prevents accidental multi-rank non-MPI launches from corrupting a shared output directory.
// Independent non-MPI processes cannot coordinate shared output.
// Require one process, or the caller explicit separate-directory policy,
// before any state or shared diagnostic files are constructed.
void check_parallel_launch_safety(const Params& p) {
#ifdef EV_USE_MPI
    (void)p;
    return;
#else
    const int world_size = detected_multi_rank_size();
    if (world_size <= 1 || p.allow_multi_rank_launch) {
        return;
    }
    throw std::runtime_error(
        "Detected a multi-rank Slurm/MPI launch with " + std::to_string(world_size)
        + " ranks, but this non-MPI executable cannot coordinate a shared output directory. "
        "Build the MPI target for distributed L-slab runs, or use one task with many threads, "
        "e.g. --ntasks=1 --cpus-per-task=$OMP_NUM_THREADS, or set EV_ALLOW_MULTI_RANK_LAUNCH=true "
        "only if every rank has a distinct output directory.");
#endif
}

/// Applies one normalized config or CLI parameter value to the Params object.
// Convert one normalized key to its declared Params field. Aliases map
// to the same field, so config and command-line vocabulary select the
// same physical controls. Unknown keys are reported with their source.
void apply_param(Params& p, const std::string& raw_key, const std::string& raw_value, const std::string& source) {
    const std::string key = normalize_key(raw_key);
    const std::string value = unquote_copy(raw_value);
    auto to_int = [&]() { return std::stoi(value); };
    auto to_double = [&]() { return std::stod(value); };
    auto to_bool = [&]() { return parse_bool_value(value, key); };

    if (key == "physics-model" || key == "model") {
        p.physics_model = value;
    } else if (key == "vp-universe" || key == "vp-background") {
        p.vp_universe = value;
    } else if (key == "vp-alpha" || key == "alpha") {
        p.vp_alpha = to_double();
    } else if (key == "vp-time-offset" || key == "vp-t0") {
        p.vp_time_offset = to_double();
    } else if (key == "vp-gravity" || key == "vp-g") {
        p.vp_gravity = to_double();
    } else if (key == "vp-poisson-source") {
        p.vp_poisson_source = value;
    } else if (key == "vp-poisson-reconstruction") {
        p.vp_poisson_reconstruction = value;
    } else if (key == "vp-interpolation" || key == "vp-transport-interpolation") {
        p.vp_interpolation = value;
    } else if (key == "kinetic-grid") {
        p.kinetic_grid = value;
    } else if (key == "kinetic-eta-grid") {
        p.kinetic_eta_grid = value;
    } else if (key == "kinetic-eta-power") {
        p.kinetic_eta_power = to_double();
    } else if (key == "kinetic-eta-join") {
        p.kinetic_eta_join = to_double();
    } else if (key == "kinetic-eta-inner-cells") {
        p.kinetic_eta_inner_cells = to_int();
    } else if (key == "kinetic-ic-quadrature-points") {
        p.kinetic_ic_quadrature_points = to_int();
    } else if (key == "vp-timestep-mode") {
        p.vp_timestep_mode = value;
    } else if (key == "vp-dynamical-eta") {
        p.vp_dynamical_eta = to_double();
    } else if (key == "vp-potential-variation-eta") {
        p.vp_potential_variation_eta = to_double();
    } else if (key == "vp-timestep-omitted-mass-fraction") {
        p.vp_timestep_omitted_mass_fraction = to_double();
    } else if (key == "vp-hamiltonian-reference") {
        p.vp_hamiltonian_reference = value;
    } else if (key == "vp-hamiltonian-reference-energy-bins") {
        p.vp_hamiltonian_reference_energy_bins = to_int();
    } else if (key == "vp-hamiltonian-reference-quadrature-points") {
        p.vp_hamiltonian_reference_quadrature_points = to_int();
    } else if (key == "vp-orbit-regularization") {
        p.vp_orbit_regularization = value;
    } else if (key == "vp-orbit-energy-tolerance") {
        p.vp_orbit_energy_tolerance = to_double();
    } else if (key == "vp-orbit-lc-relative-tolerance") {
        p.vp_orbit_lc_relative_tolerance = to_double();
    } else if (key == "vp-orbit-lc-absolute-tolerance") {
        p.vp_orbit_lc_absolute_tolerance = to_double();
    } else if (key == "vp-orbit-lc-max-steps") {
        p.vp_orbit_lc_max_steps = to_int();
    } else if (key == "vp-orbit-lc-max-refinements") {
        p.vp_orbit_lc_max_refinements = to_int();
    } else if (key == "vp-outer-inflow" || key == "vp-inflow") {
        p.vp_outer_inflow = to_bool();
    } else if (key == "vp-inflow-start-time") {
        p.vp_inflow_start_time = to_double();
    } else if (key == "vp-inflow-period") {
        p.vp_inflow_period = to_double();
    } else if (key == "vp-inflow-time-sigma") {
        p.vp_inflow_time_sigma = to_double();
    } else if (key == "vp-inflow-time-cutoff") {
        p.vp_inflow_time_cutoff = to_double();
    } else if (key == "vp-inflow-count") {
        p.vp_inflow_count = to_int();
    } else if (key == "vp-inflow-low-l-count") {
        p.vp_inflow_low_l_count = to_int();
    } else if (key == "vp-inflow-high-l-delay") {
        p.vp_inflow_high_l_delay = to_double();
    } else if (key == "vp-inflow-shell-mass") {
        p.vp_inflow_shell_mass = to_double();
    } else if (key == "vp-inflow-high-shell-mass") {
        p.vp_inflow_high_shell_mass = to_double();
    } else if (key == "vp-inflow-sigma-u") {
        p.vp_inflow_sigma_u = to_double();
    } else if (key == "vp-inflow-high-sigma-u") {
        p.vp_inflow_high_sigma_u = to_double();
    } else if (key == "vp-inflow-low-sigma-t") {
        p.vp_inflow_low_sigma_t = to_double();
    } else if (key == "vp-inflow-high-sigma-t") {
        p.vp_inflow_high_sigma_t = to_double();
    } else if (key == "vp-inflow-low-l-center") {
        p.vp_inflow_low_l_center = to_double();
    } else if (key == "vp-inflow-high-l-center") {
        p.vp_inflow_high_l_center = to_double();
    } else if (key == "vp-inflow-freefall-mass") {
        p.vp_inflow_freefall_mass = to_double();
    } else if (key == "vp-inflow-vff-factor") {
        p.vp_inflow_vff_factor = to_double();
    } else if (key == "vp-inflow-coord-speed") {
        p.vp_inflow_coord_speed = to_double();
    // The collision operator, differential kernel, material/subhalo law
    // and coefficient bath are separate choices. Their compatible physical
    // combinations are checked by setup and the selected task profile.
    } else if (key == "bp-collision-operator" || key == "collision-operator") {
        p.bp_collision_operator = value;
    } else if (key == "bp-cross-section" || key == "bp-sigma-model") {
        p.bp_cross_section = value;
    } else if (key == "bp-boltzmann-kernel" || key == "bp-differential-kernel") {
        p.bp_boltzmann_kernel = value;
    } else if (key == "bp-bath") {
        p.bp_bath = value;
    } else if (key == "bp-fp-terms") {
        p.bp_fp_terms = value;
    } else if (key == "bp-collision-flux") {
        p.bp_collision_flux = value;
    } else if (key == "bp-collision-integrator") {
        p.bp_collision_integrator = value;
    } else if (key == "bp-collision-splitting") {
        p.bp_collision_splitting = value;
    } else if (key == "bp-coeff-refresh-mode") {
        p.bp_coeff_refresh_mode = value;
    } else if (key == "bp-collision-strength") {
        p.bp_collision_strength = to_double();
    } else if (key == "bp-sigma0") {
        p.bp_sigma0 = to_double();
    } else if (key == "bp-sigma-u0" || key == "bp-u0") {
        p.bp_sigma_u0 = to_double();
    } else if (key == "bp-sigma-n" || key == "bp-n") {
        p.bp_sigma_n = to_double();
    } else if (key == "bp-subhalo-mass") {
        p.bp_subhalo_mass = to_double();
    } else if (key == "bp-subhalo-radius" || key == "bp-epsilon") {
        p.bp_subhalo_radius = to_double();
    } else if (key == "bp-coulomb-log") {
        p.bp_coulomb_log = to_double();
    } else if (key == "bp-yukawa-alpha") {
        p.bp_yukawa_alpha = to_double();
    } else if (key == "bp-yukawa-mchi") {
        p.bp_yukawa_mchi = to_double();
    } else if (key == "bp-yukawa-mphi") {
        p.bp_yukawa_mphi = to_double();
    } else if (key == "bp-temperature-floor") {
        p.bp_temperature_floor = to_double();
    } else if (key == "bp-density-floor-abs") {
        p.bp_density_floor_abs = to_double();
    } else if (key == "bp-density-floor-rel") {
        p.bp_density_floor_rel = to_double();
    } else if (key == "bp-partner-mass-tol") {
        p.bp_partner_mass_tol = to_double();
    } else if (key == "bp-bath-speed-cutoff") {
        p.bp_bath_speed_cutoff = to_double();
    } else if (key == "bp-coeff-table-points") {
        p.bp_coeff_table_points = std::max(4, to_int());
    } else if (key == "bp-coeff-speed-points") {
        p.bp_coeff_speed_points = std::max(2, to_int());
    } else if (key == "bp-coeff-mu-points") {
        p.bp_coeff_mu_points = std::max(2, to_int());
    } else if (key == "bp-coeff-refresh-every") {
        p.bp_coeff_refresh_every = std::max(1, to_int());
    } else if (key == "bp-coeff-refresh-chi") {
        p.bp_coeff_refresh_chi = to_double();
    } else if (key == "bp-coeff-refresh-time-quantile") {
        p.bp_coeff_refresh_time_quantile = to_double();
    } else if (key == "bp-collision-apply-every") {
        p.bp_collision_apply_every = std::max(1, to_int());
    } else if (key == "bp-collision-cfl") {
        p.bp_collision_cfl = to_double();
    } else if (key == "bp-collision-max-substeps") {
        p.bp_collision_max_substeps = std::max(1, to_int());
    } else if (key == "bp-collision-locality-max-ratio") {
        p.bp_collision_locality_max_ratio = to_double();
    } else if (key == "bp-collision-fail-on-nonlocal-cells") {
        p.bp_collision_fail_on_nonlocal_cells = to_bool();
    } else if (key == "bp-sts-max-stages") {
        p.bp_sts_max_stages = std::max(2, to_int());
    } else if (key == "bp-collision-l-min-rel") {
        p.bp_collision_l_min_rel = to_double();
    } else if (key == "bp-boltzmann-pair-phi-points") {
        p.bp_boltzmann_pair_phi_points = std::max(1, to_int());
    } else if (key == "bp-boltzmann-scatter-mu-points") {
        p.bp_boltzmann_scatter_mu_points = std::max(1, to_int());
    } else if (key == "bp-boltzmann-scatter-phi-points") {
        p.bp_boltzmann_scatter_phi_points = std::max(1, to_int());
    } else if (key == "bp-boltzmann-mass-tol") {
        p.bp_boltzmann_mass_tol = to_double();
    } else if (key == "bp-boltzmann-remap-max-skipped-fraction") {
        p.bp_boltzmann_remap_max_skipped_fraction = to_double();
    } else if (key == "bp-boltzmann-conserve-moments") {
        p.bp_boltzmann_conserve_moments = to_bool();
    } else if (key == "bp-reversible-candidate-radius") {
        p.bp_reversible_candidate_radius = std::max(1, to_int());
    } else if (key == "bp-reversible-max-velocity-cells") {
        p.bp_reversible_max_velocity_cells = std::max(4, to_int());
    } else if (key == "bp-reversible-projection-tolerance") {
        p.bp_reversible_projection_tolerance = to_double();
    } else if (key == "bp-reversible-entropy-floor") {
        p.bp_reversible_entropy_floor = to_double();
    } else if (key == "bp-reversible-entropy-tolerance") {
        p.bp_reversible_entropy_tolerance = to_double();
    } else if (key == "bp-reversible-active-shell-floor-fraction") {
        p.bp_reversible_active_shell_floor_fraction = to_double();
    } else if (key == "bp-reversible-explicit-fast-path") {
        p.bp_reversible_explicit_fast_path = to_bool();
    // These options bound deterministic and sampled reaction work and
    // its error estimates. They alter sampling resolution, while physical
    // collision strength remains in the separate material/cross-section keys.
    } else if (key == "bp-sampled-dense-cells") {
        p.bp_sampled_dense_cells = std::max(0, to_int());
    } else if (key == "bp-sampled-min-samples-per-batch") {
        p.bp_sampled_min_samples_per_batch = std::max(1, to_int());
    } else if (key == "bp-sampled-max-samples-per-batch") {
        p.bp_sampled_max_samples_per_batch = std::max(1, to_int());
    } else if (key == "bp-sampled-replicates-per-estimate") {
        p.bp_sampled_replicates_per_estimate = std::max(1, to_int());
    } else if (key == "bp-sampled-outgoing-angle-samples") {
        p.bp_sampled_outgoing_angle_samples = std::max(1, to_int());
    } else if (key == "bp-sampled-guided-pilot-samples") {
        p.bp_sampled_guided_pilot_samples = std::max(0, to_int());
    } else if (key == "bp-sampled-guided-pairs") {
        p.bp_sampled_guided_pairs = std::max(0, to_int());
    } else if (key == "bp-sampled-guided-fraction") {
        p.bp_sampled_guided_fraction = to_double();
    } else if (key == "bp-sampled-fixed-geometric-sequence") {
        p.bp_sampled_fixed_geometric_sequence = to_bool();
    } else if (key == "bp-sampled-reuse-operator-graph") {
        p.bp_sampled_reuse_operator_graph = to_bool();
    } else if (key == "bp-sampled-operator-refresh-l1-fraction") {
        p.bp_sampled_operator_refresh_l1_fraction = to_double();
    } else if (key == "bp-sampled-ntc-pair-thinning") {
        p.bp_sampled_ntc_pair_thinning = to_bool();
    } else if (key == "bp-sampled-ntc-pilot-samples") {
        p.bp_sampled_ntc_pilot_samples = std::max(1, to_int());
    } else if (key == "bp-sampled-ntc-quantile") {
        p.bp_sampled_ntc_quantile = to_double();
    } else if (key == "bp-sampled-energy-strata") {
        p.bp_sampled_energy_strata = std::max(1, to_int());
    } else if (key == "bp-sampled-l-strata") {
        p.bp_sampled_L_strata = std::max(1, to_int());
    } else if (key == "bp-sampled-tail-fraction") {
        p.bp_sampled_tail_fraction = to_double();
    } else if (key == "bp-sampled-stratified-fraction") {
        p.bp_sampled_stratified_fraction = to_double();
    } else if (key == "bp-sampled-equilibrium-fraction") {
        p.bp_sampled_equilibrium_fraction = to_double();
    } else if (key == "bp-sampled-phase-density-fraction") {
        p.bp_sampled_phase_density_fraction = to_double();
    } else if (key == "bp-sampled-stratum-uniform-fraction") {
        p.bp_sampled_stratum_uniform_fraction = to_double();
    } else if (key == "bp-sampled-uniform-fraction") {
        p.bp_sampled_uniform_fraction = to_double();
    } else if (key == "bp-sampled-tail-power") {
        p.bp_sampled_tail_power = to_double();
    } else if (key == "bp-sampled-rhs-tolerance") {
        p.bp_sampled_rhs_tolerance = to_double();
    } else if (key == "bp-sampled-rhs-floor-fraction") {
        p.bp_sampled_rhs_floor_fraction = to_double();
    } else if (key == "bp-sampled-heat-tolerance") {
        p.bp_sampled_heat_tolerance = to_double();
    } else if (key == "bp-sampled-heat-floor-fraction") {
        p.bp_sampled_heat_floor_fraction = to_double();
    } else if (key == "bp-sampled-activity-boost-max") {
        p.bp_sampled_activity_boost_max = to_double();
    } else if (key == "bp-sampled-fail-on-unconverged") {
        p.bp_sampled_fail_on_unconverged = to_bool();
    } else if (key == "bp-sampled-seed") {
        p.bp_sampled_seed = static_cast<std::uint64_t>(std::stoull(value));
    } else if (key == "bp-dsmc-particles") {
        p.bp_dsmc_particles = to_int();
    } else if (key == "bp-dsmc-total-mass") {
        p.bp_dsmc_total_mass = to_double();
    } else if (key == "bp-dsmc-shell-particles") {
        p.bp_dsmc_shell_particles = to_int();
    } else if (key == "bp-dsmc-min-shell-particles") {
        p.bp_dsmc_min_shell_particles = to_int();
    } else if (key == "bp-dsmc-collision-epsilon") {
        p.bp_dsmc_collision_epsilon = to_double();
    } else if (key == "bp-dsmc-gravity-eta") {
        p.bp_dsmc_gravity_eta = to_double();
    } else if (key == "bp-dsmc-seed") {
        p.bp_dsmc_seed = static_cast<std::uint64_t>(std::stoull(value));
    } else if (key == "nr") {
        p.Nr = to_int();
    } else if (key == "nw") {
        p.Nw = to_int();
    } else if (key == "nl") {
        p.NL = to_int();
    } else if (key == "r-min") {
        p.r_min = to_double();
    } else if (key == "r-max") {
        p.r_max = to_double();
    } else if (key == "radial-grid") {
        p.radial_grid = value;
    } else if (key == "custom-radial-coordinate" || key == "radial-coordinate") {
        p.custom_radial_coordinate = value;
    } else if (key == "inner-refine-radius") {
        p.inner_refine_radius = to_double();
    } else if (key == "inner-refine-cells") {
        p.inner_refine_cells = to_int();
    } else if (key == "w-max") {
        p.w_max = to_double();
    } else if (key == "l-max") {
        p.L_max = to_double();
    } else if (key == "w-stretch") {
        p.w_stretch = to_double();
    } else if (key == "w-center") {
        p.w_center = to_double();
    } else if (key == "adaptive-w") {
        p.adaptive_w = to_bool();
    } else if (key == "adaptive-w-every") {
        p.adaptive_w_every = std::max(1, to_int());
    } else if (key == "adaptive-w-trigger") {
        p.adaptive_w_trigger = to_double();
    } else if (key == "adaptive-w-margin") {
        p.adaptive_w_margin = to_double();
    } else if (key == "adaptive-w-growth") {
        p.adaptive_w_growth = to_double();
    } else if (key == "adaptive-w-max") {
        p.adaptive_w_max = to_double();
    } else if (key == "t-end") {
        p.t_end = to_double();
    } else if (key == "cfl") {
        p.cfl = to_double();
    } else if (key == "dt-max") {
        p.dt_max = to_double();
    } else if (key == "max-steps") {
        p.max_steps = to_int();
    } else if (key == "save-every") {
        p.save_every = std::max(1, to_int());
    } else if (key == "phase-every") {
        p.phase_every = std::max(0, to_int());
    } else if (key == "phase-r-stride") {
        p.phase_r_stride = std::max(1, to_int());
    } else if (key == "phase-w-stride") {
        p.phase_w_stride = std::max(1, to_int());
    } else if (key == "restart-every") {
        p.restart_every = std::max(0, to_int());
    } else if (key == "integrator") {
        p.integrator = value;
    } else if (key == "time-integrator-order" || key == "integrator-order") {
        p.time_integrator_order = to_int();
    } else if (key == "time-integrator-substeps" || key == "integrator-substeps") {
        p.time_integrator_substeps = to_int();
    } else if (key == "vp-sldg-subcells") {
        p.vp_sldg_subcells = std::max(1, to_int());
    } else if (key == "vp-sldg-eta-quadrature-points") {
        p.vp_sldg_eta_quadrature_points = to_int();
    } else if (key == "vp-sldg-geometry-tolerance") {
        p.vp_sldg_geometry_tolerance = to_double();
    } else if (key == "vp-sldg-max-area-defect") {
        p.vp_sldg_max_area_defect = to_double();
    } else if (key == "vp-sldg-max-map-defect") {
        p.vp_sldg_max_map_defect = to_double();
    } else if (key == "vp-sldg-max-refinement") {
        p.vp_sldg_max_refinement = std::max(0, to_int());
    } else if (key == "vp-sldg-max-work-per-subcell") {
        p.vp_sldg_max_work_per_subcell = to_int();
    } else if (key == "vp-sldg-fallback-mass-tolerance") {
        p.vp_sldg_fallback_mass_tolerance = to_double();
    } else if (key == "vp-sldg-initial-transport-dt") {
        p.vp_sldg_initial_transport_dt = to_double();
    } else if (key == "vp-sldg-target-fallback-fraction") {
        p.vp_sldg_target_fallback_fraction = to_double();
    } else if (key == "vp-sldg-timestep-growth") {
        p.vp_sldg_timestep_growth = to_double();
    } else if (key == "vp-sldg-timestep-shrink") {
        p.vp_sldg_timestep_shrink = to_double();
    // The unresolved-inner carrier retains its own quadrature, work and
    // return-time resolution. It belongs to the physical delayed-return
    // boundary representation, not a diagnostic output buffer.
    } else if (key == "vp-inner-carrier-quadrature-points") {
        p.vp_inner_carrier_quadrature_points = to_int();
    } else if (key == "vp-inner-carrier-max-refinement") {
        p.vp_inner_carrier_max_refinement = std::max(0, to_int());
    } else if (key == "vp-inner-carrier-max-orbit-segments") {
        p.vp_inner_carrier_max_orbit_segments = to_int();
    } else if (key == "vp-inner-carrier-max-packets") {
        p.vp_inner_carrier_max_packets = std::stoll(value);
    } else if (key == "vp-inner-carrier-mass-tolerance") {
        p.vp_inner_carrier_mass_tolerance = to_double();
    } else if (key == "vp-inner-carrier-return-time-bin-width") {
        p.vp_inner_carrier_return_time_bin_width = to_double();
    } else if (key == "vp-inner-carrier-compress-min-packets") {
        p.vp_inner_carrier_compress_min_packets = std::stoll(value);
    } else if (key == "inner-boundary") {
        p.inner_boundary = value;
    } else if (key == "outer-boundary") {
        p.outer_boundary = value;
    } else if (key == "momentum-boundary") {
        p.momentum_boundary = value;
    } else if (key == "restart-dir") {
        p.restart_dir = value;
    } else if (key == "restart-from") {
        p.restart_from = value;
    } else if (key == "threads") {
        p.threads = to_int();
    } else if (key == "allow-multi-rank-launch") {
        p.allow_multi_rank_launch = to_bool();
    } else if (key == "central-mass") {
        p.central_mass = to_double();
    } else if (key == "initial-data" || key == "initial-data-mode") {
        p.initial_data = value;
    } else if (key == "initial-data-file") {
        p.initial_data_file = value;
    } else if (key == "tracer-initial-data" || key == "tracer-initial-data-enabled") {
        p.tracer_initial_data = to_bool();
    } else if (key == "tracer-initial-data-file") {
        p.tracer_initial_data_file = value;
        p.tracer_initial_data = true;
    } else if (key == "custom-ic-normalize-mass" || key == "normalize-initial-mass") {
        p.custom_ic_normalize_mass = to_bool();
    } else if (key == "custom-ic-target-mass" || key == "target-initial-mass") {
        p.custom_ic_target_mass = to_double();
    } else if (key == "shell-initial-df") {
        p.shell_initial_df = to_bool();
    } else if (key == "r0") {
        p.R0 = to_double();
    } else if (key == "dr") {
        p.dR = to_double();
    } else if (key == "top-hat-shell") {
        p.top_hat_shell = to_bool();
        p.shell_initial_df = p.shell_initial_df || p.top_hat_shell;
    } else if (key == "top-hat-density-shell") {
        p.top_hat_density_shell = to_bool();
        p.shell_initial_df = p.shell_initial_df || p.top_hat_density_shell;
    } else if (key == "gaussian-density-shell") {
        p.gaussian_density_shell = to_bool();
        p.shell_initial_df = p.shell_initial_df || p.gaussian_density_shell;
    } else if (key == "double-gaussian-density-shell") {
        p.double_gaussian_density_shell = to_bool();
        if (p.double_gaussian_density_shell) {
            p.shell_initial_df = false;
        }
    } else if (key == "r0-2") {
        p.R0_2 = to_double();
    } else if (key == "dr-2") {
        p.dR_2 = to_double();
    } else if (key == "amp-2") {
        p.amp_2 = to_double();
    } else if (key == "w-in-2") {
        p.w_in_2 = to_double();
    } else if (key == "beta") {
        p.beta = to_double();
    } else if (key == "amp") {
        p.amp = to_double();
    } else if (key == "w-in") {
        p.w_in = to_double();
    } else if (key == "initial-metric-iters") {
        p.initial_metric_iters = std::max(0, to_int());
    } else if (key == "metric-floor") {
        p.metric_floor = to_double();
    } else if (key == "gap-stop") {
        p.gap_stop = to_double();
    } else if (key == "active-floor-rel") {
        p.active_floor_rel = to_double();
    } else if (key == "active-floor-abs") {
        p.active_floor_abs = to_double();
    } else if (key == "f-floor") {
        p.f_floor = to_double();
    } else if (key == "output-dir") {
        p.output_dir = value;
    } else {
        throw std::runtime_error("Unknown option '" + raw_key + "' in " + source + ".");
    }
}

/// Loads sectioned key-value parameters and resolves paths relative to the config file.
void load_config_file(const std::filesystem::path& path, Params& p) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open config file: " + path.string());
    }
    const std::filesystem::path config_dir = std::filesystem::absolute(path).parent_path();
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        line = trim_copy(strip_inline_comment(line));
        if (line.empty()) {
            continue;
        }
        if (line.front() == '[' && line.back() == ']') {
            continue;
        }
        std::size_t eq = line.find('=');
        if (eq == std::string::npos) {
            eq = line.find(':');
        }
        if (eq == std::string::npos) {
            throw std::runtime_error(
                "Expected 'key = value' in " + path.string() + ":" + std::to_string(line_no));
        }
        const std::string key = trim_copy(line.substr(0, eq));
        const std::string value = trim_copy(line.substr(eq + 1));
        if (key.empty() || value.empty()) {
            throw std::runtime_error(
                "Empty key or value in " + path.string() + ":" + std::to_string(line_no));
        }
        apply_param(p, key, value, path.string() + ":" + std::to_string(line_no));
    }

    // Interpret relative file paths in the directory of the loaded config.
    // This keeps a configuration and its initial data portable when invoked
    // from another working directory.
    auto resolve_path = [&](std::string& value, bool allow_empty) {
        if (value.empty()) {
            if (allow_empty) {
                return;
            }
            throw std::runtime_error("Empty path in config file: " + path.string());
        }
        std::filesystem::path candidate(value);
        if (candidate.is_relative()) {
            value = (config_dir / candidate).lexically_normal().string();
        }
    };
    resolve_path(p.initial_data_file, false);
    resolve_path(p.tracer_initial_data_file, true);
    resolve_path(p.output_dir, false);
    resolve_path(p.restart_dir, true);
    resolve_path(p.restart_from, true);
}

/// Finds input/params.ini, falling back to root params.ini.
std::filesystem::path default_config_path(const char* argv0) {
    const std::filesystem::path cwd_input_config = std::filesystem::current_path() / "input" / "params.ini";
    if (std::filesystem::exists(cwd_input_config)) {
        return cwd_input_config;
    }
    const std::filesystem::path cwd_config = std::filesystem::current_path() / "params.ini";
    if (std::filesystem::exists(cwd_config)) {
        return cwd_config;
    }
    const std::filesystem::path exe_path = std::filesystem::absolute(argv0);
    const std::filesystem::path exe_input_config = exe_path.parent_path() / "input" / "params.ini";
    if (std::filesystem::exists(exe_input_config)) {
        return exe_input_config;
    }
    const std::filesystem::path exe_config = exe_path.parent_path() / "params.ini";
    if (std::filesystem::exists(exe_config)) {
        return exe_config;
    }
    return {};
}

/// Prints command-line usage and supported options.
void print_help(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --help, -h  Show this help and exit\n"
        << "  The solver auto-loads input/params.ini from the current directory or executable directory.\n"
        << "  --config FILE, -i FILE  Read a different sectioned key=value parameter file before CLI overrides\n"
        << "  EV_CONFIG may also name the input file; CLI options override config and environment values.\n"
        << "  --physics-model einstein_vlasov|vlasov_poisson|boltzmann_poisson\n"
        << "  --vp-universe static|expanding --vp-alpha X --vp-time-offset X --vp-gravity X\n"
        << "  --vp-poisson-source density_contrast|total_density\n"
        << "  --vp-poisson-reconstruction constant_line_mass|p2_line_mass\n"
        << "  --vp-interpolation raw|hamiltonian|hamiltonian_conservative|conservative_sldg|hamiltonian_sldg|energy_shell|expanding_physical_velocity\n"
        << "  --kinetic-grid legacy_nodal|cell_average_eta\n"
        << "  --vp-timestep-mode active_cell_cfl|frank_dynamical --vp-dynamical-eta X\n"
        << "  --vp-potential-variation-eta X  Safety factor for min field/force variation time\n"
        << "  --vp-hamiltonian-reference orbit_average|none --vp-hamiltonian-reference-energy-bins N\n"
        << "  --vp-hamiltonian-reference-quadrature-points N\n"
        << "  --vp-orbit-regularization off|auto|always --vp-orbit-energy-tolerance X\n"
        << "  --vp-orbit-lc-relative-tolerance X --vp-orbit-lc-absolute-tolerance X\n"
        << "  --vp-orbit-lc-max-steps N --vp-orbit-lc-max-refinements N\n"
        << "  --vp-outer-inflow --vp-inflow-count N --vp-inflow-period X --vp-inflow-shell-mass X\n"
        << "  --vp-inflow-start-time X --vp-inflow-time-sigma X --vp-inflow-time-cutoff X\n"
        << "  --vp-inflow-low-l-count N --vp-inflow-high-l-delay X --vp-inflow-high-shell-mass X\n"
        << "  --vp-inflow-sigma-u X --vp-inflow-high-sigma-u X\n"
        << "  --vp-inflow-low-sigma-t X --vp-inflow-high-sigma-t X\n"
        << "  --vp-inflow-low-l-center X --vp-inflow-high-l-center X\n"
        << "  --vp-inflow-freefall-mass X --vp-inflow-vff-factor X --vp-inflow-coord-speed X\n"
        << "  --bp-collision-operator none|fokker_planck|landau_fp|boltzmann_exact|boltzmann_reversible|boltzmann_reversible_sampled|boltzmann_reversible_streamed|spherical_dsmc\n"
        << "  --bp-cross-section powerlaw|constant|softened_gravity|yukawa_born --bp-collision-strength X\n"
        << "  --bp-boltzmann-kernel auto|isotropic|yukawa_born  Differential kernel for Boltzmann collision backends\n"
        << "  --bp-sigma0 X --bp-u0 X --bp-n X --bp-bath instantaneous_df|local_maxwellian|global_maxwellian\n"
        << "  --bp-sigma-u0 X --bp-sigma-n X --bp-subhalo-mass X --bp-subhalo-radius X\n"
        << "  --bp-coulomb-log X --bp-yukawa-alpha X --bp-yukawa-mchi X --bp-yukawa-mphi X\n"
        << "  --bp-temperature-floor X --bp-density-floor-abs X --bp-density-floor-rel X\n"
        << "  --bp-bath-speed-cutoff X\n"
        << "  --bp-fp-terms full|vr_only|L_only|no_mixed|no_azimuth|no_drift|drift_only|Drr_only|DLL_only|Daa_only\n"
        << "  --bp-collision-flux centered|chang_cooper --bp-collision-integrator explicit|sts_rkl2\n"
        << "  --bp-collision-splitting after_vlasov|strang --bp-coeff-refresh-mode steps|collision_time\n"
        << "  --bp-partner-mass-tol X  # optional instantaneous-DF tail pruning; 0 is exact\n"
        << "  --bp-coeff-table-points N --bp-coeff-speed-points N --bp-coeff-mu-points N\n"
        << "  --bp-coeff-refresh-every N  Reuse instantaneous-DF coefficients for N steps; default 1\n"
        << "  --bp-coeff-refresh-chi X --bp-coeff-refresh-time-quantile X --bp-sts-max-stages N\n"
        << "  --bp-collision-l-min-rel X  Ignore angular collision flux below L/Lmax=X; default 0\n"
        << "  --bp-collision-apply-every N  Apply collisions every N advection steps; default 1\n"
        << "  --bp-collision-cfl X --bp-collision-max-substeps N\n"
        << "  --bp-collision-locality-max-ratio X --bp-collision-fail-on-nonlocal-cells true|false\n"
        << "  --bp-boltzmann-pair-phi-points N --bp-boltzmann-scatter-mu-points N --bp-boltzmann-scatter-phi-points N\n"
        << "  --bp-boltzmann-mass-tol X --bp-boltzmann-remap-max-skipped-fraction X\n"
        << "  --bp-boltzmann-conserve-moments true|false\n"
        << "  --bp-reversible-candidate-radius N\n"
        << "  --bp-reversible-max-velocity-cells N --bp-reversible-projection-tolerance X\n"
        << "  --bp-reversible-entropy-floor X --bp-reversible-entropy-tolerance X\n"
        << "  --bp-reversible-active-shell-floor-fraction X\n"
        << "  --bp-reversible-explicit-fast-path true|false\n"
        << "  --bp-sampled-dense-cells N --bp-sampled-min-samples-per-batch N --bp-sampled-max-samples-per-batch N\n"
        << "  --bp-sampled-guided-pilot-samples N --bp-sampled-guided-pairs N --bp-sampled-guided-fraction X\n"
        << "  --bp-sampled-fixed-geometric-sequence true|false --bp-sampled-reuse-operator-graph true|false\n"
        << "  --bp-sampled-operator-refresh-l1-fraction X\n"
        << "  --bp-sampled-ntc-pair-thinning true|false --bp-sampled-ntc-pilot-samples N\n"
        << "  --bp-sampled-ntc-quantile X\n"
        << "  --bp-sampled-replicates-per-estimate N\n"
        << "  --bp-sampled-outgoing-angle-samples N\n"
        << "  --bp-sampled-energy-strata N --bp-sampled-l-strata N --bp-sampled-tail-fraction X\n"
        << "  --bp-sampled-stratified-fraction X --bp-sampled-equilibrium-fraction X\n"
        << "  --bp-sampled-phase-density-fraction X\n"
        << "  --bp-sampled-stratum-uniform-fraction X\n"
        << "  --bp-sampled-uniform-fraction X --bp-sampled-tail-power X\n"
        << "  --bp-sampled-rhs-tolerance X --bp-sampled-rhs-floor-fraction X --bp-sampled-heat-tolerance X\n"
        << "  --bp-sampled-heat-floor-fraction X --bp-sampled-activity-boost-max X\n"
        << "  --bp-sampled-fail-on-unconverged true|false --bp-sampled-seed N\n"
        << "  --bp-dsmc-particles N --bp-dsmc-total-mass X --bp-dsmc-shell-particles N\n"
        << "  --bp-dsmc-min-shell-particles N\n"
        << "  --bp-dsmc-collision-epsilon X --bp-dsmc-gravity-eta X --bp-dsmc-seed N\n"
        << "  --Nr N --Nw N --NL N\n"
        << "  --radial-grid log|linear|piecewise-linear --custom-radial-coordinate log|linear --r-min X --r-max X\n"
        << "  --inner-refine-radius X --inner-refine-cells N --w-max X --L-max X --w-stretch X --w-center X\n"
        << "  --adaptive-w --adaptive-w-every N --adaptive-w-trigger X --adaptive-w-margin X --adaptive-w-growth X --adaptive-w-max X\n"
        << "  --t-end X --dt-max X --cfl X --save-every N --phase-every N --max-steps N\n"
        << "  --phase-r-stride N --phase-w-stride N --restart-every N --restart-dir DIR --restart-from PATH\n"
        << "  --integrator semi_lagrangian_kick_drift_kick|semi_lagrangian_rk|semi_lagrangian_adams_bashforth|conservative_finite_volume\n"
        << "  --time-integrator-order N  RK2-RK4 or extrapolated midpoint order 5..32; AB: 2..12\n"
        << "  --time-integrator-substeps N  optional characteristic substep/extrapolation multiplier\n"
        << "  --vp-sldg-subcells N --vp-sldg-geometry-tolerance X --vp-sldg-eta-quadrature-points N\n"
        << "  --vp-timestep-omitted-mass-fraction X\n"
        << "  --kinetic-eta-grid uniform|power|composite_power --kinetic-eta-power X\n"
        << "  --kinetic-eta-join X --kinetic-eta-inner-cells N --kinetic-ic-quadrature-points N\n"
        << "  --vp-sldg-max-area-defect X --vp-sldg-max-map-defect X --vp-sldg-max-refinement N\n"
        << "  --vp-sldg-max-work-per-subcell N\n"
        << "  --vp-sldg-fallback-mass-tolerance X\n"
        << "  --vp-sldg-initial-transport-dt X --vp-sldg-target-fallback-fraction X\n"
        << "  --vp-sldg-timestep-growth X --vp-sldg-timestep-shrink X\n"
        << "  --vp-inner-carrier-quadrature-points N --vp-inner-carrier-max-refinement N\n"
        << "  --vp-inner-carrier-max-orbit-segments N\n"
        << "  --vp-inner-carrier-max-packets N --vp-inner-carrier-mass-tolerance X\n"
        << "  --vp-inner-carrier-return-time-bin-width X\n"
        << "  --vp-inner-carrier-compress-min-packets N\n"
        << "  --inner-boundary reflecting|delayed_return|origin_parity --outer-boundary outflow_vacuum|reflecting|vp_outer_inflow --momentum-boundary outflow_vacuum\n"
        << "  --threads N  (0 means auto from EV_THREADS/OMP_NUM_THREADS/SLURM_CPUS_PER_TASK)\n"
        << "  --allow-multi-rank-launch  Non-MPI build only: bypass duplicate-rank output guard\n"
        << "  --initial-data analytic|binary_rwl --initial-data-file PATH\n"
        << "  --tracer-initial-data-file PATH\n"
        << "  --custom-ic-normalize-mass --custom-ic-target-mass X\n"
        << "  --central-mass X --beta X --amp X --w-in X --initial-metric-iters N\n"
        << "  --metric-floor X --gap-stop X\n"
        << "  --shell-initial-df --R0 X --dR X --top-hat-shell --top-hat-density-shell --gaussian-density-shell\n"
        << "  --double-gaussian-density-shell --R0-2 X --dR-2 X --amp-2 X --w-in-2 X\n"
        << "  --active-floor-rel X --active-floor-abs X --f-floor X --output-dir DIR --restart-dir DIR\n";
}

/// Builds final runtime parameters from defaults, config files, environment, and CLI overrides.
// The first argument pass loads configuration before any ordinary
// CLI overrides, independent of where --config appears in the argument
// list. Help exits before creating a physical simulation state.
Params parse_args(int argc, char** argv) {
    Params p;

    bool saw_config = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            std::exit(0);
        }
        if (arg == "--config" || arg == "-i" || arg == "--input" || arg == "--input-file") {
            saw_config = true;
            load_config_file(need(arg), p);
        }
    }
    if (!saw_config) {
        const char* env_config = getenv_nonempty("EV_CONFIG");
        const std::filesystem::path config_path =
            (env_config != nullptr) ? std::filesystem::path(env_config) : default_config_path(argv[0]);
        if (!config_path.empty()) {
            load_config_file(config_path, p);
        }
    }

    // Track explicit output and restart destinations separately. A chosen
    // output directory can supply its own default restart subdirectory
    // without replacing an explicitly chosen restart destination.
    bool output_dir_overridden = environment_overrides_output_dir();
    bool restart_dir_overridden = environment_overrides_restart_dir();
    apply_environment_overrides(p);

    // The second pass applies actual CLI values after the environment.
    // Each option requiring a value consumes the next argument explicitly.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else if (arg == "--config" || arg == "-i" || arg == "--input" || arg == "--input-file") {
            (void)need(arg);
        } else if (arg == "--physics-model" || arg == "--model") {
            p.physics_model = need(arg);
        } else if (arg == "--vp-universe" || arg == "--vp-background") {
            p.vp_universe = need(arg);
        } else if (arg == "--vp-alpha" || arg == "--alpha") {
            p.vp_alpha = std::stod(need(arg));
        } else if (arg == "--vp-time-offset" || arg == "--vp-t0") {
            p.vp_time_offset = std::stod(need(arg));
        } else if (arg == "--vp-gravity" || arg == "--vp-g") {
            p.vp_gravity = std::stod(need(arg));
        } else if (arg == "--vp-poisson-source") {
            p.vp_poisson_source = need(arg);
        } else if (arg == "--vp-poisson-reconstruction") {
            p.vp_poisson_reconstruction = need(arg);
        } else if (arg == "--vp-interpolation" || arg == "--vp-transport-interpolation") {
            p.vp_interpolation = need(arg);
        } else if (arg == "--kinetic-grid") {
            p.kinetic_grid = need(arg);
        } else if (arg == "--kinetic-eta-grid") {
            p.kinetic_eta_grid = need(arg);
        } else if (arg == "--kinetic-eta-power") {
            p.kinetic_eta_power = std::stod(need(arg));
        } else if (arg == "--kinetic-eta-join") {
            p.kinetic_eta_join = std::stod(need(arg));
        } else if (arg == "--kinetic-eta-inner-cells") {
            p.kinetic_eta_inner_cells = std::stoi(need(arg));
        } else if (arg == "--kinetic-ic-quadrature-points") {
            p.kinetic_ic_quadrature_points = std::stoi(need(arg));
        } else if (arg == "--vp-timestep-mode") {
            p.vp_timestep_mode = need(arg);
        } else if (arg == "--vp-dynamical-eta") {
            p.vp_dynamical_eta = std::stod(need(arg));
        } else if (arg == "--vp-potential-variation-eta") {
            p.vp_potential_variation_eta = std::stod(need(arg));
        } else if (arg == "--vp-timestep-omitted-mass-fraction") {
            p.vp_timestep_omitted_mass_fraction = std::stod(need(arg));
        } else if (arg == "--vp-hamiltonian-reference") {
            p.vp_hamiltonian_reference = need(arg);
        } else if (arg == "--vp-hamiltonian-reference-energy-bins") {
            p.vp_hamiltonian_reference_energy_bins = std::stoi(need(arg));
        } else if (arg == "--vp-hamiltonian-reference-quadrature-points") {
            p.vp_hamiltonian_reference_quadrature_points = std::stoi(need(arg));
        } else if (arg == "--vp-orbit-regularization") {
            p.vp_orbit_regularization = need(arg);
        } else if (arg == "--vp-orbit-energy-tolerance") {
            p.vp_orbit_energy_tolerance = std::stod(need(arg));
        } else if (arg == "--vp-orbit-lc-relative-tolerance") {
            p.vp_orbit_lc_relative_tolerance = std::stod(need(arg));
        } else if (arg == "--vp-orbit-lc-absolute-tolerance") {
            p.vp_orbit_lc_absolute_tolerance = std::stod(need(arg));
        } else if (arg == "--vp-orbit-lc-max-steps") {
            p.vp_orbit_lc_max_steps = std::stoi(need(arg));
        } else if (arg == "--vp-orbit-lc-max-refinements") {
            p.vp_orbit_lc_max_refinements = std::stoi(need(arg));
        } else if (arg == "--vp-outer-inflow" || arg == "--vp-inflow") {
            p.vp_outer_inflow = true;
        } else if (arg == "--vp-inflow-start-time") {
            p.vp_inflow_start_time = std::stod(need(arg));
        } else if (arg == "--vp-inflow-period") {
            p.vp_inflow_period = std::stod(need(arg));
        } else if (arg == "--vp-inflow-time-sigma") {
            p.vp_inflow_time_sigma = std::stod(need(arg));
        } else if (arg == "--vp-inflow-time-cutoff") {
            p.vp_inflow_time_cutoff = std::stod(need(arg));
        } else if (arg == "--vp-inflow-count") {
            p.vp_inflow_count = std::stoi(need(arg));
        } else if (arg == "--vp-inflow-low-l-count") {
            p.vp_inflow_low_l_count = std::stoi(need(arg));
        } else if (arg == "--vp-inflow-high-l-delay") {
            p.vp_inflow_high_l_delay = std::stod(need(arg));
        } else if (arg == "--vp-inflow-shell-mass") {
            p.vp_inflow_shell_mass = std::stod(need(arg));
        } else if (arg == "--vp-inflow-high-shell-mass") {
            p.vp_inflow_high_shell_mass = std::stod(need(arg));
        } else if (arg == "--vp-inflow-sigma-u") {
            p.vp_inflow_sigma_u = std::stod(need(arg));
        } else if (arg == "--vp-inflow-high-sigma-u") {
            p.vp_inflow_high_sigma_u = std::stod(need(arg));
        } else if (arg == "--vp-inflow-low-sigma-t") {
            p.vp_inflow_low_sigma_t = std::stod(need(arg));
        } else if (arg == "--vp-inflow-high-sigma-t") {
            p.vp_inflow_high_sigma_t = std::stod(need(arg));
        } else if (arg == "--vp-inflow-low-l-center") {
            p.vp_inflow_low_l_center = std::stod(need(arg));
        } else if (arg == "--vp-inflow-high-l-center") {
            p.vp_inflow_high_l_center = std::stod(need(arg));
        } else if (arg == "--vp-inflow-freefall-mass") {
            p.vp_inflow_freefall_mass = std::stod(need(arg));
        } else if (arg == "--vp-inflow-vff-factor") {
            p.vp_inflow_vff_factor = std::stod(need(arg));
        } else if (arg == "--vp-inflow-coord-speed") {
            p.vp_inflow_coord_speed = std::stod(need(arg));
        } else if (arg == "--bp-collision-operator" || arg == "--collision-operator") {
            p.bp_collision_operator = need(arg);
        } else if (arg == "--bp-cross-section" || arg == "--bp-sigma-model") {
            p.bp_cross_section = need(arg);
        } else if (arg == "--bp-boltzmann-kernel" || arg == "--bp-differential-kernel") {
            p.bp_boltzmann_kernel = need(arg);
        } else if (arg == "--bp-bath") {
            p.bp_bath = need(arg);
        } else if (arg == "--bp-fp-terms") {
            p.bp_fp_terms = need(arg);
        } else if (arg == "--bp-collision-flux") {
            p.bp_collision_flux = need(arg);
        } else if (arg == "--bp-collision-integrator") {
            p.bp_collision_integrator = need(arg);
        } else if (arg == "--bp-collision-splitting") {
            p.bp_collision_splitting = need(arg);
        } else if (arg == "--bp-coeff-refresh-mode") {
            p.bp_coeff_refresh_mode = need(arg);
        } else if (arg == "--bp-collision-strength") {
            p.bp_collision_strength = std::stod(need(arg));
        } else if (arg == "--bp-sigma0") {
            p.bp_sigma0 = std::stod(need(arg));
        } else if (arg == "--bp-sigma-u0" || arg == "--bp-u0") {
            p.bp_sigma_u0 = std::stod(need(arg));
        } else if (arg == "--bp-sigma-n" || arg == "--bp-n") {
            p.bp_sigma_n = std::stod(need(arg));
        } else if (arg == "--bp-subhalo-mass") {
            p.bp_subhalo_mass = std::stod(need(arg));
        } else if (arg == "--bp-subhalo-radius" || arg == "--bp-epsilon") {
            p.bp_subhalo_radius = std::stod(need(arg));
        } else if (arg == "--bp-coulomb-log") {
            p.bp_coulomb_log = std::stod(need(arg));
        } else if (arg == "--bp-yukawa-alpha") {
            p.bp_yukawa_alpha = std::stod(need(arg));
        } else if (arg == "--bp-yukawa-mchi") {
            p.bp_yukawa_mchi = std::stod(need(arg));
        } else if (arg == "--bp-yukawa-mphi") {
            p.bp_yukawa_mphi = std::stod(need(arg));
        } else if (arg == "--bp-temperature-floor") {
            p.bp_temperature_floor = std::stod(need(arg));
        } else if (arg == "--bp-density-floor-abs") {
            p.bp_density_floor_abs = std::stod(need(arg));
        } else if (arg == "--bp-density-floor-rel") {
            p.bp_density_floor_rel = std::stod(need(arg));
        } else if (arg == "--bp-partner-mass-tol") {
            p.bp_partner_mass_tol = std::stod(need(arg));
        } else if (arg == "--bp-bath-speed-cutoff") {
            p.bp_bath_speed_cutoff = std::stod(need(arg));
        } else if (arg == "--bp-coeff-table-points") {
            p.bp_coeff_table_points = std::max(4, std::stoi(need(arg)));
        } else if (arg == "--bp-coeff-speed-points") {
            p.bp_coeff_speed_points = std::max(2, std::stoi(need(arg)));
        } else if (arg == "--bp-coeff-mu-points") {
            p.bp_coeff_mu_points = std::max(2, std::stoi(need(arg)));
        } else if (arg == "--bp-coeff-refresh-every") {
            p.bp_coeff_refresh_every = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--bp-coeff-refresh-chi") {
            p.bp_coeff_refresh_chi = std::stod(need(arg));
        } else if (arg == "--bp-coeff-refresh-time-quantile") {
            p.bp_coeff_refresh_time_quantile = std::stod(need(arg));
        } else if (arg == "--bp-collision-apply-every") {
            p.bp_collision_apply_every = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--bp-collision-cfl") {
            p.bp_collision_cfl = std::stod(need(arg));
        } else if (arg == "--bp-collision-max-substeps") {
            p.bp_collision_max_substeps = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--bp-collision-locality-max-ratio") {
            p.bp_collision_locality_max_ratio = std::stod(need(arg));
        } else if (arg == "--bp-collision-fail-on-nonlocal-cells") {
            p.bp_collision_fail_on_nonlocal_cells =
                parse_bool_value(need(arg), arg);
        } else if (arg == "--bp-sts-max-stages") {
            p.bp_sts_max_stages = std::max(2, std::stoi(need(arg)));
        } else if (arg == "--bp-collision-l-min-rel") {
            p.bp_collision_l_min_rel = std::stod(need(arg));
        } else if (arg == "--bp-boltzmann-pair-phi-points") {
            p.bp_boltzmann_pair_phi_points = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--bp-boltzmann-scatter-mu-points") {
            p.bp_boltzmann_scatter_mu_points = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--bp-boltzmann-scatter-phi-points") {
            p.bp_boltzmann_scatter_phi_points = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--bp-boltzmann-mass-tol") {
            p.bp_boltzmann_mass_tol = std::stod(need(arg));
        } else if (arg == "--bp-boltzmann-remap-max-skipped-fraction") {
            p.bp_boltzmann_remap_max_skipped_fraction = std::stod(need(arg));
        } else if (arg == "--bp-boltzmann-conserve-moments") {
            p.bp_boltzmann_conserve_moments = parse_bool_value(need(arg), arg);
        } else if (arg == "--bp-reversible-candidate-radius") {
            p.bp_reversible_candidate_radius = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--bp-reversible-max-velocity-cells") {
            p.bp_reversible_max_velocity_cells = std::max(4, std::stoi(need(arg)));
        } else if (arg == "--bp-reversible-projection-tolerance") {
            p.bp_reversible_projection_tolerance = std::stod(need(arg));
        } else if (arg == "--bp-reversible-entropy-floor") {
            p.bp_reversible_entropy_floor = std::stod(need(arg));
        } else if (arg == "--bp-reversible-entropy-tolerance") {
            p.bp_reversible_entropy_tolerance = std::stod(need(arg));
        } else if (arg == "--bp-reversible-active-shell-floor-fraction") {
            p.bp_reversible_active_shell_floor_fraction = std::stod(need(arg));
        } else if (arg == "--bp-reversible-explicit-fast-path") {
            p.bp_reversible_explicit_fast_path =
                parse_bool_value(need(arg), arg);
        } else if (arg == "--bp-sampled-dense-cells") {
            p.bp_sampled_dense_cells = std::max(0, std::stoi(need(arg)));
        } else if (arg == "--bp-sampled-min-samples-per-batch") {
            p.bp_sampled_min_samples_per_batch = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--bp-sampled-max-samples-per-batch") {
            p.bp_sampled_max_samples_per_batch = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--bp-sampled-replicates-per-estimate") {
            p.bp_sampled_replicates_per_estimate =
                std::max(1, std::stoi(need(arg)));
        } else if (arg == "--bp-sampled-outgoing-angle-samples") {
            p.bp_sampled_outgoing_angle_samples = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--bp-sampled-guided-pilot-samples") {
            p.bp_sampled_guided_pilot_samples = std::max(0, std::stoi(need(arg)));
        } else if (arg == "--bp-sampled-guided-pairs") {
            p.bp_sampled_guided_pairs = std::max(0, std::stoi(need(arg)));
        } else if (arg == "--bp-sampled-guided-fraction") {
            p.bp_sampled_guided_fraction = std::stod(need(arg));
        } else if (arg == "--bp-sampled-fixed-geometric-sequence") {
            p.bp_sampled_fixed_geometric_sequence =
                parse_bool_value(need(arg), arg);
        } else if (arg == "--bp-sampled-reuse-operator-graph") {
            p.bp_sampled_reuse_operator_graph =
                parse_bool_value(need(arg), arg);
        } else if (arg == "--bp-sampled-operator-refresh-l1-fraction") {
            p.bp_sampled_operator_refresh_l1_fraction =
                std::stod(need(arg));
        } else if (arg == "--bp-sampled-ntc-pair-thinning") {
            p.bp_sampled_ntc_pair_thinning =
                parse_bool_value(need(arg), arg);
        } else if (arg == "--bp-sampled-ntc-pilot-samples") {
            p.bp_sampled_ntc_pilot_samples =
                std::max(1, std::stoi(need(arg)));
        } else if (arg == "--bp-sampled-ntc-quantile") {
            p.bp_sampled_ntc_quantile = std::stod(need(arg));
        } else if (arg == "--bp-sampled-energy-strata") {
            p.bp_sampled_energy_strata = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--bp-sampled-l-strata") {
            p.bp_sampled_L_strata = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--bp-sampled-tail-fraction") {
            p.bp_sampled_tail_fraction = std::stod(need(arg));
        } else if (arg == "--bp-sampled-stratified-fraction") {
            p.bp_sampled_stratified_fraction = std::stod(need(arg));
        } else if (arg == "--bp-sampled-equilibrium-fraction") {
            p.bp_sampled_equilibrium_fraction = std::stod(need(arg));
        } else if (arg == "--bp-sampled-phase-density-fraction") {
            p.bp_sampled_phase_density_fraction = std::stod(need(arg));
        } else if (arg == "--bp-sampled-stratum-uniform-fraction") {
            p.bp_sampled_stratum_uniform_fraction = std::stod(need(arg));
        } else if (arg == "--bp-sampled-uniform-fraction") {
            p.bp_sampled_uniform_fraction = std::stod(need(arg));
        } else if (arg == "--bp-sampled-tail-power") {
            p.bp_sampled_tail_power = std::stod(need(arg));
        } else if (arg == "--bp-sampled-rhs-tolerance") {
            p.bp_sampled_rhs_tolerance = std::stod(need(arg));
        } else if (arg == "--bp-sampled-rhs-floor-fraction") {
            p.bp_sampled_rhs_floor_fraction = std::stod(need(arg));
        } else if (arg == "--bp-sampled-heat-tolerance") {
            p.bp_sampled_heat_tolerance = std::stod(need(arg));
        } else if (arg == "--bp-sampled-heat-floor-fraction") {
            p.bp_sampled_heat_floor_fraction = std::stod(need(arg));
        } else if (arg == "--bp-sampled-activity-boost-max") {
            p.bp_sampled_activity_boost_max = std::stod(need(arg));
        } else if (arg == "--bp-sampled-fail-on-unconverged") {
            p.bp_sampled_fail_on_unconverged = parse_bool_value(need(arg), arg);
        } else if (arg == "--bp-sampled-seed") {
            p.bp_sampled_seed = static_cast<std::uint64_t>(std::stoull(need(arg)));
        } else if (arg == "--bp-dsmc-particles") {
            p.bp_dsmc_particles = std::stoi(need(arg));
        } else if (arg == "--bp-dsmc-total-mass") {
            p.bp_dsmc_total_mass = std::stod(need(arg));
        } else if (arg == "--bp-dsmc-shell-particles") {
            p.bp_dsmc_shell_particles = std::stoi(need(arg));
        } else if (arg == "--bp-dsmc-min-shell-particles") {
            p.bp_dsmc_min_shell_particles = std::stoi(need(arg));
        } else if (arg == "--bp-dsmc-collision-epsilon") {
            p.bp_dsmc_collision_epsilon = std::stod(need(arg));
        } else if (arg == "--bp-dsmc-gravity-eta") {
            p.bp_dsmc_gravity_eta = std::stod(need(arg));
        } else if (arg == "--bp-dsmc-seed") {
            p.bp_dsmc_seed = static_cast<std::uint64_t>(std::stoull(need(arg)));
        } else if (arg == "--Nr") {
            p.Nr = std::stoi(need(arg));
        } else if (arg == "--Nw") {
            p.Nw = std::stoi(need(arg));
        } else if (arg == "--NL") {
            p.NL = std::stoi(need(arg));
        } else if (arg == "--r-min") {
            p.r_min = std::stod(need(arg));
        } else if (arg == "--r-max") {
            p.r_max = std::stod(need(arg));
        } else if (arg == "--radial-grid") {
            p.radial_grid = need(arg);
        } else if (arg == "--custom-radial-coordinate" || arg == "--radial-coordinate") {
            p.custom_radial_coordinate = need(arg);
        } else if (arg == "--inner-refine-radius") {
            p.inner_refine_radius = std::stod(need(arg));
        } else if (arg == "--inner-refine-cells") {
            p.inner_refine_cells = std::stoi(need(arg));
        } else if (arg == "--w-max") {
            p.w_max = std::stod(need(arg));
        } else if (arg == "--L-max") {
            p.L_max = std::stod(need(arg));
        } else if (arg == "--w-stretch") {
            p.w_stretch = std::stod(need(arg));
        } else if (arg == "--w-center") {
            p.w_center = std::stod(need(arg));
        } else if (arg == "--adaptive-w") {
            p.adaptive_w = true;
        } else if (arg == "--adaptive-w-every") {
            p.adaptive_w_every = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--adaptive-w-trigger") {
            p.adaptive_w_trigger = std::stod(need(arg));
        } else if (arg == "--adaptive-w-margin") {
            p.adaptive_w_margin = std::stod(need(arg));
        } else if (arg == "--adaptive-w-growth") {
            p.adaptive_w_growth = std::stod(need(arg));
        } else if (arg == "--adaptive-w-max") {
            p.adaptive_w_max = std::stod(need(arg));
        } else if (arg == "--t-end") {
            p.t_end = std::stod(need(arg));
        } else if (arg == "--cfl") {
            p.cfl = std::stod(need(arg));
        } else if (arg == "--dt-max") {
            p.dt_max = std::stod(need(arg));
        } else if (arg == "--max-steps") {
            p.max_steps = std::stoi(need(arg));
        } else if (arg == "--save-every") {
            p.save_every = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--phase-every") {
            p.phase_every = std::max(0, std::stoi(need(arg)));
        } else if (arg == "--phase-r-stride") {
            p.phase_r_stride = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--phase-w-stride") {
            p.phase_w_stride = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--restart-every") {
            p.restart_every = std::max(0, std::stoi(need(arg)));
        } else if (arg == "--restart-dir") {
            p.restart_dir = need(arg);
            restart_dir_overridden = true;
        } else if (arg == "--restart-from") {
            p.restart_from = need(arg);
        } else if (arg == "--integrator") {
            p.integrator = need(arg);
        } else if (arg == "--time-integrator-order" || arg == "--integrator-order") {
            p.time_integrator_order = std::stoi(need(arg));
        } else if (arg == "--time-integrator-substeps" || arg == "--integrator-substeps") {
            p.time_integrator_substeps = std::stoi(need(arg));
        } else if (arg == "--vp-sldg-subcells") {
            p.vp_sldg_subcells = std::max(1, std::stoi(need(arg)));
        } else if (arg == "--vp-sldg-eta-quadrature-points") {
            p.vp_sldg_eta_quadrature_points = std::stoi(need(arg));
        } else if (arg == "--vp-sldg-geometry-tolerance") {
            p.vp_sldg_geometry_tolerance = std::stod(need(arg));
        } else if (arg == "--vp-sldg-max-area-defect") {
            p.vp_sldg_max_area_defect = std::stod(need(arg));
        } else if (arg == "--vp-sldg-max-map-defect") {
            p.vp_sldg_max_map_defect = std::stod(need(arg));
        } else if (arg == "--vp-sldg-max-refinement") {
            p.vp_sldg_max_refinement = std::max(0, std::stoi(need(arg)));
        } else if (arg == "--vp-sldg-max-work-per-subcell") {
            p.vp_sldg_max_work_per_subcell = std::stoi(need(arg));
        } else if (arg == "--vp-sldg-fallback-mass-tolerance") {
            p.vp_sldg_fallback_mass_tolerance = std::stod(need(arg));
        } else if (arg == "--vp-sldg-initial-transport-dt") {
            p.vp_sldg_initial_transport_dt = std::stod(need(arg));
        } else if (arg == "--vp-sldg-target-fallback-fraction") {
            p.vp_sldg_target_fallback_fraction = std::stod(need(arg));
        } else if (arg == "--vp-sldg-timestep-growth") {
            p.vp_sldg_timestep_growth = std::stod(need(arg));
        } else if (arg == "--vp-sldg-timestep-shrink") {
            p.vp_sldg_timestep_shrink = std::stod(need(arg));
        } else if (arg == "--vp-inner-carrier-quadrature-points") {
            p.vp_inner_carrier_quadrature_points = std::stoi(need(arg));
        } else if (arg == "--vp-inner-carrier-max-refinement") {
            p.vp_inner_carrier_max_refinement =
                std::max(0, std::stoi(need(arg)));
        } else if (arg == "--vp-inner-carrier-max-orbit-segments") {
            p.vp_inner_carrier_max_orbit_segments = std::stoi(need(arg));
        } else if (arg == "--vp-inner-carrier-max-packets") {
            p.vp_inner_carrier_max_packets = std::stoll(need(arg));
        } else if (arg == "--vp-inner-carrier-mass-tolerance") {
            p.vp_inner_carrier_mass_tolerance = std::stod(need(arg));
        } else if (arg == "--vp-inner-carrier-return-time-bin-width") {
            p.vp_inner_carrier_return_time_bin_width = std::stod(need(arg));
        } else if (arg == "--vp-inner-carrier-compress-min-packets") {
            p.vp_inner_carrier_compress_min_packets = std::stoll(need(arg));
        } else if (arg == "--inner-boundary") {
            p.inner_boundary = need(arg);
        } else if (arg == "--outer-boundary") {
            p.outer_boundary = need(arg);
        } else if (arg == "--momentum-boundary") {
            p.momentum_boundary = need(arg);
        } else if (arg == "--threads") {
            p.threads = std::stoi(need(arg));
        } else if (arg == "--allow-multi-rank-launch") {
            p.allow_multi_rank_launch = true;
        } else if (arg == "--central-mass") {
            p.central_mass = std::stod(need(arg));
        } else if (arg == "--initial-data") {
            p.initial_data = need(arg);
        } else if (arg == "--initial-data-file") {
            p.initial_data_file = need(arg);
        } else if (arg == "--tracer-initial-data-file") {
            p.tracer_initial_data_file = need(arg);
            p.tracer_initial_data = true;
        } else if (arg == "--custom-ic-normalize-mass") {
            p.custom_ic_normalize_mass = true;
        } else if (arg == "--custom-ic-target-mass") {
            p.custom_ic_target_mass = std::stod(need(arg));
        } else if (arg == "--shell-initial-df") {
            p.shell_initial_df = true;
        } else if (arg == "--R0") {
            p.R0 = std::stod(need(arg));
        } else if (arg == "--dR") {
            p.dR = std::stod(need(arg));
        } else if (arg == "--top-hat-shell") {
            p.top_hat_shell = true;
            p.shell_initial_df = true;
        } else if (arg == "--top-hat-density-shell") {
            p.shell_initial_df = true;
            p.top_hat_density_shell = true;
        } else if (arg == "--gaussian-density-shell") {
            p.shell_initial_df = true;
            p.gaussian_density_shell = true;
        } else if (arg == "--double-gaussian-density-shell") {
            p.shell_initial_df = false;
            p.double_gaussian_density_shell = true;
        } else if (arg == "--R0-2") {
            p.R0_2 = std::stod(need(arg));
        } else if (arg == "--dR-2") {
            p.dR_2 = std::stod(need(arg));
        } else if (arg == "--amp-2") {
            p.amp_2 = std::stod(need(arg));
        } else if (arg == "--w-in-2") {
            p.w_in_2 = std::stod(need(arg));
        } else if (arg == "--beta") {
            p.beta = std::stod(need(arg));
        } else if (arg == "--amp") {
            p.amp = std::stod(need(arg));
        } else if (arg == "--w-in") {
            p.w_in = std::stod(need(arg));
        } else if (arg == "--initial-metric-iters") {
            p.initial_metric_iters = std::max(0, std::stoi(need(arg)));
        } else if (arg == "--metric-floor") {
            p.metric_floor = std::stod(need(arg));
        } else if (arg == "--gap-stop") {
            p.gap_stop = std::stod(need(arg));
        } else if (arg == "--active-floor-rel") {
            p.active_floor_rel = std::stod(need(arg));
        } else if (arg == "--active-floor-abs") {
            p.active_floor_abs = std::stod(need(arg));
        } else if (arg == "--f-floor") {
            p.f_floor = std::stod(need(arg));
        } else if (arg == "--output-dir") {
            p.output_dir = need(arg);
            output_dir_overridden = true;
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }
    // Finish path defaults and parallel launch checks only after every
    // parameter source has been merged. The returned record therefore
    // contains the actual values used by state construction.
    if (output_dir_overridden && !restart_dir_overridden) {
        p.restart_dir = (std::filesystem::path(p.output_dir) / "restarts").lexically_normal().string();
    }
    finalize_parallel_runtime(p);
    check_parallel_launch_safety(p);
    return p;
}

}  // namespace ev
