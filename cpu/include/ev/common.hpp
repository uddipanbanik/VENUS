#pragma once

// Define the shared physical parameters and data records of the CPU solvers.
// The Newtonian family evolves a spherical distribution with Poisson gravity
// and optional collisions; Einstein--Vlasov uses relativistic stress and metric
// constraints. Params separates their model choices, initial/boundary data,
// phase-space resolution, accuracy controls and output settings.
// The records below carry moments, metric fields, boundary budgets and collision
// diagnostics between modules. They define data meanings rather than perform
// a field solve or advance physical time.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef EV_USE_MPI
#include <mpi.h>
#endif


namespace ev {


constexpr double PI = 3.141592653589793238462643383279502884;

/// Compares a fixed-width, NUL-terminated binary identifier without reading
/// beyond its field.  Constructing std::string(field) is undefined for a
/// malformed field containing no NUL byte, which is exactly the case binary
/// readers must reject safely.
template <std::size_t N>
inline bool fixed_binary_identifier_equals(
    const char (&field)[N],
    const char* expected) {
    const std::size_t expected_size = std::char_traits<char>::length(expected);
    return expected_size < N
        && std::equal(expected, expected + expected_size, field)
        && field[expected_size] == '\0';
}

/// Flushes a CSV/text stream and reports disk/quota failures immediately.
inline void flush_output_checked(std::ostream& stream, const std::string& label) {
    stream.flush();
    if (!stream) {
        throw std::runtime_error("Failed to flush output stream: " + label);
    }
}

// Runtime parameters.  The names intentionally match the long command-line
// options and the keys accepted by the config-file parser below.
/// Collects all user-configurable runtime parameters for grids, physics, IO, and parallelism.
struct Params {
    // Physics model.  The default keeps the relativistic Einstein-Vlasov
    // solver.  "vlasov_poisson" uses the weak-field Newtonian/FLRW module.
    // "boltzmann_poisson" is the same spherical VP dynamics with an optional
    // local collision operator in velocity space.
    std::string physics_model = "einstein_vlasov";
    std::string vp_universe = "static";
    double vp_alpha = 2.0 / 3.0;
    double vp_time_offset = 1.0;
    double vp_gravity = 1.0;
    std::string vp_poisson_source = "total_density";
    std::string vp_poisson_reconstruction = "constant_line_mass"; // constant_line_mass|p2_line_mass
    std::string vp_interpolation = "raw"; // raw|hamiltonian|hamiltonian_conservative|conservative_sldg|hamiltonian_sldg|energy_shell|expanding_physical_velocity
    // The nodal representation samples f(r,w,L). The cell-average path
    // uses canonical eta=L^2/2 and its dr dw deta measure. This choice
    // determines how initial data, transport and collision quadrature agree.
    std::string kinetic_grid = "legacy_nodal"; // legacy_nodal|cell_average_eta
    std::string kinetic_eta_grid = "power"; // uniform|power|composite_power
    double kinetic_eta_power = 3.0;
    double kinetic_eta_join = 0.0;
    int kinetic_eta_inner_cells = 0;
    int kinetic_ic_quadrature_points = 4;
    // Clock controls are numerical accuracy choices. The activity/mass
    // budget concerns which cells set a global clock, while the state and
    // its complete gravitational source remain represented by their solver.
    std::string vp_timestep_mode = "active_cell_cfl"; // active_cell_cfl|frank_dynamical
    double vp_dynamical_eta = 0.01;
    double vp_potential_variation_eta = 0.1;
    double vp_timestep_omitted_mass_fraction = 1.0e-12;
    std::string vp_hamiltonian_reference = "orbit_average"; // orbit_average|none
    int vp_hamiltonian_reference_energy_bins = 512;
    int vp_hamiltonian_reference_quadrature_points = 4;
    std::string vp_orbit_regularization = "auto"; // off|auto|always
    double vp_orbit_energy_tolerance = 1.0e-9;
    double vp_orbit_lc_relative_tolerance = 1.0e-9;
    double vp_orbit_lc_absolute_tolerance = 1.0e-12;
    int vp_orbit_lc_max_steps = 2048;
    int vp_orbit_lc_max_refinements = 2;

    // Optional VP-only right-boundary inflow.  The boundary condition injects
    // Gaussian time pulses at r_max; early pulses can be cold/low-L and later
    // pulses can switch to a high-angular-momentum distribution.
    bool vp_outer_inflow = false;
    double vp_inflow_start_time = 0.0;
    double vp_inflow_period = 1.0;
    double vp_inflow_time_sigma = 0.15;
    double vp_inflow_time_cutoff = 5.0;
    int vp_inflow_count = 0;
    int vp_inflow_low_l_count = 10;
    double vp_inflow_high_l_delay = 0.0;
    double vp_inflow_shell_mass = 1.0e-3;
    double vp_inflow_high_shell_mass = -1.0;
    double vp_inflow_sigma_u = 0.04;
    double vp_inflow_high_sigma_u = -1.0;
    double vp_inflow_low_sigma_t = 0.0015;
    double vp_inflow_high_sigma_t = 0.035;
    double vp_inflow_low_l_center = 0.0;
    double vp_inflow_high_l_center = 0.0;
    double vp_inflow_freefall_mass = 0.37;
    double vp_inflow_vff_factor = 1.0;
    double vp_inflow_coord_speed = 0.0;

    // Optional Boltzmann-Poisson collision module.  The Fokker-Planck/Landau
    // path uses a transfer cross section sigma_T(u).  The finite-angle
    // Boltzmann path below instead uses an explicit differential kernel.
    std::string bp_collision_operator = "none";       // none|fokker_planck|landau_fp|boltzmann_exact|boltzmann_reversible|boltzmann_reversible_sampled|boltzmann_reversible_streamed|spherical_dsmc
    std::string bp_cross_section = "powerlaw";        // powerlaw|constant|softened_gravity|yukawa_born
    std::string bp_boltzmann_kernel = "auto";         // auto|isotropic|yukawa_born
    std::string bp_bath = "local_maxwellian";         // instantaneous_df|local_maxwellian|global_maxwellian
    std::string bp_fp_terms = "full";                 // diagnostic FP term mask; default keeps all terms
    std::string bp_collision_flux = "chang_cooper";   // centered|chang_cooper
    std::string bp_collision_integrator = "explicit"; // explicit|sts_rkl2
    std::string bp_collision_splitting = "strang";    // after_vlasov|strang
    std::string bp_coeff_refresh_mode = "steps";      // steps|collision_time
    double bp_collision_strength = 1.0;
    double bp_sigma0 = 0.0;
    double bp_sigma_u0 = 1.0;
    double bp_sigma_n = 4.0;
    double bp_subhalo_mass = 1.0e-3;
    double bp_subhalo_radius = 0.05;
    double bp_coulomb_log = 3.0;
    // Yukawa inputs set the speed scale mphi/mchi and the zero-speed value
    // 4*pi*alpha^2*mchi^2/mphi^4. Grid collision readers interpret that value
    // as area; material WPIC/SphDSMC readers interpret it as area per mass.
    // Convert the physical law to the selected module's units before setting
    // alpha; it is an effective amplitude after that conversion.
    double bp_yukawa_alpha = 1.0e-2;
    double bp_yukawa_mchi = 1.0;
    double bp_yukawa_mphi = 1.0;
    double bp_temperature_floor = 1.0e-8;
    double bp_density_floor_abs = 0.0;
    double bp_density_floor_rel = 1.0e-14;
    double bp_partner_mass_tol = 0.0;
    double bp_bath_speed_cutoff = 6.0;
    int bp_coeff_table_points = 32;
    int bp_coeff_speed_points = 20;
    int bp_coeff_mu_points = 16;
    int bp_coeff_refresh_every = 1;
    double bp_coeff_refresh_chi = 0.1;
    double bp_coeff_refresh_time_quantile = 0.0;
    int bp_collision_apply_every = 1;
    double bp_collision_cfl = 0.15;
    int bp_collision_max_substeps = 200;
    double bp_collision_locality_max_ratio = 1.0;
    bool bp_collision_fail_on_nonlocal_cells = true;
    int bp_sts_max_stages = 80;
    double bp_collision_l_min_rel = 0.0;
    // Finite-angle quadrature resolves the relative tangential orientation
    // and the outgoing solid angle. Its skipped-activity and moment-repair
    // controls describe the finite velocity-domain projection.
    int bp_boltzmann_pair_phi_points = 4;
    int bp_boltzmann_scatter_mu_points = 4;
    int bp_boltzmann_scatter_phi_points = 4;
    double bp_boltzmann_mass_tol = 0.0;
    double bp_boltzmann_remap_max_skipped_fraction = 1.0e-6;
    bool bp_boltzmann_conserve_moments = true;
    int bp_reversible_candidate_radius = 1;
    int bp_reversible_max_velocity_cells = 256;
    double bp_reversible_projection_tolerance = 1.0e-11;
    double bp_reversible_entropy_floor = 1.0e-300;
    double bp_reversible_entropy_tolerance = 1.0e-12;
    double bp_reversible_active_shell_floor_fraction = 1.0e-12;
    bool bp_reversible_explicit_fast_path = true;
    // The sampled reversible method treats a selected dense subset
    // deterministically and samples the remaining reaction channels. Proposal
    // mixtures and replicate counts control variance, not physical coupling.
    int bp_sampled_dense_cells = 16;
    int bp_sampled_min_samples_per_batch = 16384;
    int bp_sampled_max_samples_per_batch = 131072;
    int bp_sampled_replicates_per_estimate = 1;
    int bp_sampled_outgoing_angle_samples = 4;
    int bp_sampled_guided_pilot_samples = 2048;
    int bp_sampled_guided_pairs = 512;
    double bp_sampled_guided_fraction = 0.5;
    bool bp_sampled_fixed_geometric_sequence = true;
    bool bp_sampled_reuse_operator_graph = true;
    double bp_sampled_operator_refresh_l1_fraction = 0.02;
    bool bp_sampled_ntc_pair_thinning = false;
    int bp_sampled_ntc_pilot_samples = 512;
    double bp_sampled_ntc_quantile = 0.10;
    int bp_sampled_energy_strata = 4;
    int bp_sampled_L_strata = 3;
    double bp_sampled_tail_fraction = 0.25;
    double bp_sampled_stratified_fraction = 0.20;
    double bp_sampled_equilibrium_fraction = 0.10;
    double bp_sampled_phase_density_fraction = 0.25;
    double bp_sampled_stratum_uniform_fraction = 0.20;
    double bp_sampled_uniform_fraction = 0.02;
    double bp_sampled_tail_power = 1.5;
    double bp_sampled_rhs_tolerance = 0.01;
    double bp_sampled_rhs_floor_fraction = 0.1;
    double bp_sampled_heat_tolerance = 0.01;
    double bp_sampled_heat_floor_fraction = 0.1;
    double bp_sampled_activity_boost_max = 4.0;
    bool bp_sampled_fail_on_unconverged = true;
    std::uint64_t bp_sampled_seed = UINT64_C(1442695040888963407);
    // Persistent spherical DSMC instead carries particles between force
    // and collision steps. Shell population controls its spatial collision
    // averaging; total represented mass and cross-section units belong
    // to that particle formulation.
    int bp_dsmc_particles = 1000000;
    double bp_dsmc_total_mass = 0.0;
    int bp_dsmc_shell_particles = 4096;
    int bp_dsmc_min_shell_particles = 32;
    double bp_dsmc_collision_epsilon = 1.0;
    double bp_dsmc_gravity_eta = 0.01;
    std::uint64_t bp_dsmc_seed = UINT64_C(5489);

    // Phase-space resolution.
    int Nr = 260;
    int Nw = 240;
    int NL = 48;

    // Radial grid and momentum-domain controls.
    double r_min = 1.0e-3;
    double r_max = 60.0;
    std::string radial_grid = "log";
    std::string custom_radial_coordinate = "linear";
    double inner_refine_radius = 0.1;
    int inner_refine_cells = 96;
    double w_max = 8.0;
    double L_max = 6.0;
    double w_stretch = 2.5;
    double w_center = 0.0;
    // Adaptive velocity bounds can move or enlarge the stored w mesh.
    // Their trigger and margin are numerical support controls; the remap
    // must account for the distribution and its invariant budgets.
    bool adaptive_w = false;
    int adaptive_w_every = 20;
    double adaptive_w_trigger = 0.65;
    double adaptive_w_margin = 1.8;
    double adaptive_w_growth = 1.6;
    double adaptive_w_max = 0.0;

    // Time stepping and output cadence.
    double t_end = 20.0;
    double cfl = 0.35;
    double dt_max = 0.04;
    int max_steps = 200000;
    // These output cadences count accepted steps. They describe saved
    // observations and restarts separately from the coordinate-time endpoint
    // and the physical accuracy limits on each step.
    int save_every = 20;
    int phase_every = 0;
    int phase_r_stride = 1;
    int phase_w_stride = 1;
    int restart_every = 0;
    std::string integrator = "semi_lagrangian_kick_drift_kick";
    int time_integrator_order = 2;
    int time_integrator_substeps = 1;
    int vp_sldg_subcells = 2;
    int vp_sldg_eta_quadrature_points = 2;
    double vp_sldg_geometry_tolerance = 1.0e-10;
    double vp_sldg_max_area_defect = 1.0e-4;
    double vp_sldg_max_map_defect = 1.0e-2;
    int vp_sldg_max_refinement = 30;
    int vp_sldg_max_work_per_subcell = 131072;
    double vp_sldg_fallback_mass_tolerance = 1.0e-9;
    double vp_sldg_initial_transport_dt = 0.0;
    double vp_sldg_target_fallback_fraction = 0.25;
    double vp_sldg_timestep_growth = 1.5;
    double vp_sldg_timestep_shrink = 0.5;
    // Delayed inner return retains mass and its physical flight through
    // the unresolved central region. These quadrature/work controls resolve
    // that carried distribution and its return times rather than deleting it.
    int vp_inner_carrier_quadrature_points = 4;
    int vp_inner_carrier_max_refinement = 6;
    int vp_inner_carrier_max_orbit_segments = 16;
    std::int64_t vp_inner_carrier_max_packets = 2000000;
    double vp_inner_carrier_mass_tolerance = 1.0e-12;
    double vp_inner_carrier_return_time_bin_width = 0.0;
    std::int64_t vp_inner_carrier_compress_min_packets = 200000;
    std::string inner_boundary = "reflecting";
    std::string outer_boundary = "outflow_vacuum";
    std::string momentum_boundary = "outflow_vacuum";

    // A prescribed unresolved central source is an inner boundary datum.
    // It differs from the matter mass obtained by depositing the resolved
    // distribution and must be counted once in the field reconstruction.
    double central_mass = 0.0;

    // Initial-data source.  "analytic" keeps the built-in shell/boundary-seed
    // families.  "binary_rwl" reads a user supplied f(r,w,L) file whose header
    // carries the r, w, and L grids.
    std::string initial_data = "analytic";
    std::string initial_data_file = "initial_data/example_shell_rwl.bin";
    bool custom_ic_normalize_mass = false;
    double custom_ic_target_mass = 1.0e-2;
    bool tracer_initial_data = false;
    std::string tracer_initial_data_file;

    // Initial data.  The shell options use a Maxwell-Juttner-like momentum
    // seed evaluated at R0 so the radial profile is not accidentally baked in.
    bool shell_initial_df = false;
    double R0 = 5.0;
    double dR = 1.0;
    bool top_hat_shell = false;
    bool top_hat_density_shell = false;
    bool gaussian_density_shell = false;
    bool double_gaussian_density_shell = false;
    double beta = 4.0;
    double amp = 1.0e-2;
    double w_in = -0.6;
    double R0_2 = 10.0;
    double dR_2 = 1.0;
    double amp_2 = 1.0e-2;
    double w_in_2 = -1.0;
    int initial_metric_iters = 4;

    // The metric evaluation floor protects arithmetic near a chart limit;
    // gap_stop is its separate stopping threshold. Activity and f floors
    // have their own distribution/support roles and are recorded explicitly.
    double metric_floor = 1.0e-10;
    double gap_stop = 2.0e-5;
    double active_floor_rel = 1.0e-12;
    double active_floor_abs = 0.0;
    double f_floor = 0.0;

    // Parallelism and file layout.  threads=0 means derive from EV_THREADS,
    // OMP_NUM_THREADS, SLURM_CPUS_PER_TASK, or the local hardware.
    int threads = 0;
    bool allow_multi_rank_launch = false;

    std::string output_dir = "output/venus_modular";
    std::string restart_dir;
    std::string restart_from;
};

/// Holds metadata and values for an EVICRWL1 binary initial-data file on an (r,w,L) grid.
struct InitialDataRWL {
    int Nr = 0;
    int Nw = 0;
    int NL = 0;
    std::vector<double> r;
    std::vector<double> w;
    std::vector<double> L;
    std::vector<double> f;
};

/// Stores density, radial/tangential pressure, and radial current moments.
// In the relativistic module rho is energy density, p is radial
// pressure, pt is the sum of two tangential pressures and j is radial
// energy current. Each module supplies their stated quadrature measure.
struct Moments {
    std::vector<double> rho;
    std::vector<double> p;
    std::vector<double> pt;
    std::vector<double> j;
};

/// Stores the spherical metric fields and derivatives reconstructed from the Einstein constraints.
struct Metric {
    std::vector<double> m;
    std::vector<double> lambda;
    std::vector<double> mu;
    std::vector<double> lambda_r;
    std::vector<double> mu_r;
    std::vector<double> lambda_t;
    std::vector<double> gap;
    // Optional finite-volume face reconstruction.  The nodal
    // semi-Lagrangian solver leaves these arrays empty; the Einstein-Vlasov
    // WPIC solver fills them so characteristics see the regular origin and
    // the same near-horizon compactness used by the radial constraint.
    std::vector<double> face_radius;
    std::vector<double> face_m;
    std::vector<double> face_lambda;
    std::vector<double> face_mu;
    std::vector<double> face_mu_r;
    std::vector<double> face_lambda_t;
    std::vector<double> face_gap;
    double min_gap = 1.0;
};

/// Accumulates estimated phase-space mass that leaves the numerical domain during one step.
struct EscapeStats {
    double outer_mass = 0.0;
    double w_low_mass = 0.0;
    double w_high_mass = 0.0;
    double total_mass = 0.0;
    long long outer_cells = 0;
    long long w_low_cells = 0;
    long long w_high_cells = 0;
};

/// Stores phase-space RMS magnitudes of the Vlasov and collision operators.
struct OperatorDiagnostics {
    double vlasov_rms = 0.0;
    double collision_rms = 0.0;
    double collision_over_vlasov = 0.0;
    double vlasov_rms_r_lt_1 = 0.0;
    double collision_rms_r_lt_1 = 0.0;
    double collision_over_vlasov_r_lt_1 = 0.0;
    double vlasov_rms_r_1_3 = 0.0;
    double collision_rms_r_1_3 = 0.0;
    double collision_over_vlasov_r_1_3 = 0.0;
    double vlasov_rms_r_ge_3 = 0.0;
    double collision_rms_r_ge_3 = 0.0;
    double collision_over_vlasov_r_ge_3 = 0.0;
};

/// Diagnostics from one local collision source update.
// Retain collision accuracy, activity, work and locality diagnostics
// separately. An invariant residual, omitted channel fraction and
// Monte Carlo error estimate answer different acceptance questions.
struct BPCollisionStepInfo {
    int substeps = 0;
    int explicit_substeps = 0;
    int sts_cycles = 0;
    double collision_dt = 0.0;
    double explicit_dt_limit = 0.0;
    double collision_time_estimate = 0.0;
    double coeff_refresh_dt = 0.0;
    double max_D_rr = 0.0;
    double max_D_LL = 0.0;
    double max_abs_D_rL = 0.0;
    double max_abs_D_r = 0.0;
    double max_abs_D_L = 0.0;
    double remap_attempted_event_density = 0.0;
    double remap_skipped_event_density = 0.0;
    double remap_skipped_w_event_density = 0.0;
    double remap_skipped_L_event_density = 0.0;
    double max_shell_remap_skipped_fraction = 0.0;
    double remap_skipped_domain_event_density = 0.0;
    double remap_skipped_projection_event_density = 0.0;
    double remap_skipped_inactive_event_density = 0.0;
    double maximum_projection_error = 0.0;
    std::uint64_t sampled_dense_reactions = 0;
    std::uint64_t sampled_complement_reactions = 0;
    std::uint64_t sampled_complement_draws = 0;
    std::uint64_t sampled_retained_collision_draws = 0;
    std::uint64_t sampled_projection_cache_hits = 0;
    std::uint64_t sampled_projection_cache_misses = 0;
    int sampled_reused_operator_shells = 0;
    double sampled_max_operator_reference_l1 = 0.0;
    int sampled_ntc_fallback_shells = 0;
    int sampled_min_samples_per_batch = 0;
    int sampled_max_samples_per_batch = 0;
    int sampled_max_adaptive_rounds = 0;
    int sampled_unconverged_shells = 0;
    int sampled_inactive_shells = 0;
    double sampled_max_rhs_relative_error = 0.0;
    double sampled_max_rhs_gross_error = 0.0;
    double sampled_max_heat_relative_error = 0.0;
    double sampled_max_heat_gross_error = 0.0;
    double sampled_max_energy_invariant_error = 0.0;
    double sampled_context_cpu_seconds = 0.0;
    double sampled_estimate_cpu_seconds = 0.0;
    double sampled_materialize_cpu_seconds = 0.0;
    double sampled_advance_cpu_seconds = 0.0;
    double sampled_max_shell_seconds = 0.0;
    double minimum_mean_free_path = std::numeric_limits<double>::infinity();
    double maximum_radial_cell_width_over_mean_free_path = 0.0;
    int nonlocal_collision_cells = 0;
    std::uint64_t reversible_reaction_updates = 0;
    std::uint64_t reversible_explicit_fast_path_fallbacks = 0;
    std::uint64_t reversible_implicit_reaction_updates = 0;
    std::uint64_t reversible_implicit_iterations = 0;
};

/// Combines diagnostics from consecutive collision stages into one step ledger.
///
/// Extensive counters and timings are summed, restrictive timestep/mean-free-
/// path limits are minimized, and worst-case errors are maximized.  Keeping the
/// complete merge next to BPCollisionStepInfo prevents a new backend diagnostic
/// from silently disappearing when Strang splitting combines its two half steps.
inline void merge_bp_collision_step_info(
    BPCollisionStepInfo& total,
    const BPCollisionStepInfo& part) {
    total.substeps += part.substeps;
    total.explicit_substeps += part.explicit_substeps;
    total.sts_cycles += part.sts_cycles;
    total.collision_dt = std::max(total.collision_dt, part.collision_dt);
    total.explicit_dt_limit =
        (total.explicit_dt_limit > 0.0 && part.explicit_dt_limit > 0.0)
        ? std::min(total.explicit_dt_limit, part.explicit_dt_limit)
        : std::max(total.explicit_dt_limit, part.explicit_dt_limit);
    total.collision_time_estimate =
        (total.collision_time_estimate > 0.0
         && part.collision_time_estimate > 0.0)
        ? std::min(
            total.collision_time_estimate,
            part.collision_time_estimate)
        : std::max(
            total.collision_time_estimate,
            part.collision_time_estimate);
    total.coeff_refresh_dt = std::max(
        total.coeff_refresh_dt, part.coeff_refresh_dt);
    total.max_D_rr = std::max(total.max_D_rr, part.max_D_rr);
    total.max_D_LL = std::max(total.max_D_LL, part.max_D_LL);
    total.max_abs_D_rL = std::max(total.max_abs_D_rL, part.max_abs_D_rL);
    total.max_abs_D_r = std::max(total.max_abs_D_r, part.max_abs_D_r);
    total.max_abs_D_L = std::max(total.max_abs_D_L, part.max_abs_D_L);
    // Attempted and skipped activities are extensive across split stages.
    // Add their actual contributions before forming any omission fraction.
    total.remap_attempted_event_density +=
        part.remap_attempted_event_density;
    total.remap_skipped_event_density += part.remap_skipped_event_density;
    total.remap_skipped_w_event_density +=
        part.remap_skipped_w_event_density;
    total.remap_skipped_L_event_density +=
        part.remap_skipped_L_event_density;
    total.max_shell_remap_skipped_fraction = std::max(
        total.max_shell_remap_skipped_fraction,
        part.max_shell_remap_skipped_fraction);
    total.remap_skipped_domain_event_density +=
        part.remap_skipped_domain_event_density;
    total.remap_skipped_projection_event_density +=
        part.remap_skipped_projection_event_density;
    total.remap_skipped_inactive_event_density +=
        part.remap_skipped_inactive_event_density;
    total.maximum_projection_error = std::max(
        total.maximum_projection_error, part.maximum_projection_error);
    total.sampled_dense_reactions += part.sampled_dense_reactions;
    total.sampled_complement_reactions += part.sampled_complement_reactions;
    total.sampled_complement_draws += part.sampled_complement_draws;
    total.sampled_retained_collision_draws +=
        part.sampled_retained_collision_draws;
    total.sampled_projection_cache_hits +=
        part.sampled_projection_cache_hits;
    total.sampled_projection_cache_misses +=
        part.sampled_projection_cache_misses;
    total.sampled_reused_operator_shells +=
        part.sampled_reused_operator_shells;
    total.sampled_max_operator_reference_l1 = std::max(
        total.sampled_max_operator_reference_l1,
        part.sampled_max_operator_reference_l1);
    total.sampled_ntc_fallback_shells += part.sampled_ntc_fallback_shells;
    if (part.sampled_min_samples_per_batch > 0) {
        total.sampled_min_samples_per_batch =
            total.sampled_min_samples_per_batch > 0
            ? std::min(
                total.sampled_min_samples_per_batch,
                part.sampled_min_samples_per_batch)
            : part.sampled_min_samples_per_batch;
    }
    total.sampled_max_samples_per_batch = std::max(
        total.sampled_max_samples_per_batch,
        part.sampled_max_samples_per_batch);
    total.sampled_max_adaptive_rounds = std::max(
        total.sampled_max_adaptive_rounds,
        part.sampled_max_adaptive_rounds);
    total.sampled_unconverged_shells += part.sampled_unconverged_shells;
    total.sampled_inactive_shells += part.sampled_inactive_shells;
    total.sampled_max_rhs_relative_error = std::max(
        total.sampled_max_rhs_relative_error,
        part.sampled_max_rhs_relative_error);
    total.sampled_max_rhs_gross_error = std::max(
        total.sampled_max_rhs_gross_error,
        part.sampled_max_rhs_gross_error);
    total.sampled_max_heat_relative_error = std::max(
        total.sampled_max_heat_relative_error,
        part.sampled_max_heat_relative_error);
    total.sampled_max_heat_gross_error = std::max(
        total.sampled_max_heat_gross_error,
        part.sampled_max_heat_gross_error);
    total.sampled_max_energy_invariant_error = std::max(
        total.sampled_max_energy_invariant_error,
        part.sampled_max_energy_invariant_error);
    total.sampled_context_cpu_seconds += part.sampled_context_cpu_seconds;
    total.sampled_estimate_cpu_seconds += part.sampled_estimate_cpu_seconds;
    total.sampled_materialize_cpu_seconds +=
        part.sampled_materialize_cpu_seconds;
    total.sampled_advance_cpu_seconds += part.sampled_advance_cpu_seconds;
    total.sampled_max_shell_seconds = std::max(
        total.sampled_max_shell_seconds, part.sampled_max_shell_seconds);
    // A restrictive length is combined by a minimum; a locality ratio
    // or error is combined by a maximum. This retains the worst physical
    // cell across successive collision stages.
    total.minimum_mean_free_path = std::min(
        total.minimum_mean_free_path, part.minimum_mean_free_path);
    total.maximum_radial_cell_width_over_mean_free_path = std::max(
        total.maximum_radial_cell_width_over_mean_free_path,
        part.maximum_radial_cell_width_over_mean_free_path);
    total.nonlocal_collision_cells = std::max(
        total.nonlocal_collision_cells, part.nonlocal_collision_cells);
    total.reversible_reaction_updates += part.reversible_reaction_updates;
    total.reversible_explicit_fast_path_fallbacks +=
        part.reversible_explicit_fast_path_fallbacks;
    total.reversible_implicit_reaction_updates +=
        part.reversible_implicit_reaction_updates;
    total.reversible_implicit_iterations +=
        part.reversible_implicit_iterations;
}

/// Holds metric values interpolated at one characteristic footpoint.
struct MetricSample {
    double mu = 0.0;
    double lambda = 0.0;
    double mu_r = 0.0;
    double lambda_t = 0.0;
};

/// Records whether MPI is active and identifies this rank within MPI_COMM_WORLD.
struct MpiContext {
    bool enabled = false;
    int rank = 0;
    int size = 1;
};

/// Describes the contiguous angular-momentum slab owned by one MPI rank.
struct LDecomposition {
    int global_NL = 0;
    int offset = 0;
    int count = 0;
};

/// True when the weak-field Vlasov-Poisson family is selected.
// These predicates describe configured equation/algorithm families.
// Both Newtonian VP and BP share the Poisson/transport field path;
// additional predicates select their actual local collision operator.
inline bool using_vlasov_poisson(const Params& p) {
    return p.physics_model == "vlasov_poisson" || p.physics_model == "boltzmann_poisson";
}

/// True when the weak-field branch should apply a local collision operator.
inline bool using_boltzmann_poisson(const Params& p) {
    return p.physics_model == "boltzmann_poisson"
        || (using_vlasov_poisson(p) && p.bp_collision_operator != "none");
}

/// True when the runnable Fokker-Planck/Landau collision update is enabled.
inline bool using_bp_fokker_planck(const Params& p) {
    return using_boltzmann_poisson(p)
        && (p.bp_collision_operator == "fokker_planck" || p.bp_collision_operator == "landau_fp");
}

/// True when the local finite-angle Boltzmann collision update is enabled.
inline bool using_bp_finite_angle_boltzmann(const Params& p) {
    return using_boltzmann_poisson(p) && p.bp_collision_operator == "boltzmann_exact";
}

/// True for the entropy-dissipative finite-angle collision bracket.
inline bool using_bp_reversible_boltzmann(const Params& p) {
    return using_boltzmann_poisson(p) && p.bp_collision_operator == "boltzmann_reversible";
}

/// True for the hybrid dense-plus-importance-sampled reversible bracket.
inline bool using_bp_sampled_reversible_boltzmann(const Params& p) {
    return using_boltzmann_poisson(p)
        && p.bp_collision_operator == "boltzmann_reversible_sampled";
}

/// True for the compact streamed sampled reversible Boltzmann bracket.
/// It avoids the full complement graph but materializes a compressed graph
/// from the retained sampled channels before applying the reaction update.
inline bool using_bp_streamed_reversible_boltzmann(const Params& p) {
    return using_boltzmann_poisson(p)
        && p.bp_collision_operator == "boltzmann_reversible_streamed";
}

/// True when persistent spherical DSMC markers are the evolution state.
/// Fresh samples are usually equal mass, but restarts and collisions may carry
/// an unequal positive mass spectrum.
inline bool using_bp_spherical_dsmc(const Params& p) {
    return using_boltzmann_poisson(p) && p.bp_collision_operator == "spherical_dsmc";
}

/// True for the mass-conservative forward Hamiltonian remap.
inline bool using_conservative_hamiltonian_transport(const Params& p) {
    return using_vlasov_poisson(p) && p.vp_interpolation == "hamiltonian_conservative";
}

/// True for the cell-integrated conservative semi-Lagrangian transport.
inline bool using_conservative_sldg_transport(const Params& p) {
    return using_vlasov_poisson(p) && p.vp_interpolation == "conservative_sldg";
}

/// True for conservative SLDG with a local Hamiltonian/phase reconstruction.
inline bool using_hamiltonian_sldg_transport(const Params& p) {
    return using_vlasov_poisson(p) && p.vp_interpolation == "hamiltonian_sldg";
}

/// Uses one remap per potential-variation macrostep in a static universe.
inline bool using_static_hamiltonian_potential_macrostep(const Params& p) {
    return using_hamiltonian_sldg_transport(p) && p.vp_universe == "static";
}

/// True for either cell-integrated conservative SLDG reconstruction.
inline bool using_any_sldg_transport(const Params& p) {
    return using_conservative_sldg_transport(p)
        || using_hamiltonian_sldg_transport(p);
}

/// True when trajectories crossing r_min spend their physical flight time in
/// the unresolved central control volume before returning through the face.
inline bool using_delayed_inner_return(const Params& p) {
    return using_vlasov_poisson(p) && p.inner_boundary == "delayed_return";
}

/// True when f stores cell averages in (r,w,eta=L^2/2).
inline bool using_cell_average_eta_grid(const Params& p) {
    return p.kinetic_grid == "cell_average_eta";
}

/// True when radial quadrature must use the conservative physical cell widths.
inline bool using_conservative_phase_space_cells(const Params& p) {
    return using_conservative_hamiltonian_transport(p)
        || using_any_sldg_transport(p);
}

/// True when VP/BP characteristics may retry a stiff orbit in regularized coordinates.
inline bool using_vp_orbit_regularization(const Params& p) {
    return using_vlasov_poisson(p) && p.vp_orbit_regularization != "off";
}

/// True when any runnable local BP collision update is enabled.
inline bool using_bp_collision_operator(const Params& p) {
    return using_bp_fokker_planck(p)
        || using_bp_finite_angle_boltzmann(p)
        || using_bp_reversible_boltzmann(p)
        || using_bp_sampled_reversible_boltzmann(p)
        || using_bp_streamed_reversible_boltzmann(p)
        || using_bp_spherical_dsmc(p);
}

/// Rejects MPI decompositions that do not implement the selected operator.
///
/// Collisionless EV/VP transport decomposes the angular-momentum axis across
/// MPI ranks.  Every BP collision backend, including the persistent-marker
/// SphDSMC backend, currently requires the complete local collision state and
/// therefore must run as one MPI rank with shared-memory threads.
inline void validate_mpi_execution_mode(const Params& p, const MpiContext& mpi) {
    if (using_bp_collision_operator(p) && mpi.size != 1) {
        throw std::runtime_error(
            "Boltzmann-Poisson collision and SphDSMC backends are OpenMP-only: "
            "launch exactly one MPI task and place parallelism in threads.");
    }
    if (mpi.size > p.NL) {
        throw std::runtime_error(
            "MPI L-slab decomposition requires mpi_size <= NL. "
            "Use no more MPI ranks than angular-momentum cells, or move extra cores to OpenMP "
            "threads per rank with --cpus-per-task/OMP_NUM_THREADS.");
    }
}

/// True when the VP branch should include Hubble advection and drag.
inline bool using_expanding_vp(const Params& p) {
    return using_vlasov_poisson(p) && p.vp_universe == "expanding";
}

/// True when the VP outer boundary should inject a scheduled inflow source.
inline bool using_vp_outer_inflow(const Params& p) {
    return using_vlasov_poisson(p)
        && (p.vp_outer_inflow || p.outer_boundary == "vp_outer_inflow");
}

/// True for the second-order characteristic kick-drift-kick tracer.
inline bool using_kick_drift_kick_integrator(const Params& p) {
    return p.integrator == "semi_lagrangian_kick_drift_kick";
}

/// True for explicit Runge-Kutta characteristic tracing.
inline bool using_rk_characteristic_integrator(const Params& p) {
    return p.integrator == "semi_lagrangian_rk"
        || p.integrator == "runge_kutta"
        || p.integrator == "rk";
}

/// True for the configured Adams--Bashforth characteristic tracer.
inline bool using_adams_bashforth_characteristic_integrator(const Params& p) {
    return p.integrator == "semi_lagrangian_adams_bashforth"
        || p.integrator == "adams_bashforth";
}

/// True when the integrator is one of the semi-Lagrangian characteristic tracers.
inline bool using_characteristic_integrator(const Params& p) {
    return using_kick_drift_kick_integrator(p)
        || using_rk_characteristic_integrator(p)
        || using_adams_bashforth_characteristic_integrator(p);
}
}  // namespace ev
