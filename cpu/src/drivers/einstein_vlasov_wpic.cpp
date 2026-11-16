// Run spherical Einstein--Vlasov particle evolution in either Schwarzschild
// coordinate time or ingoing Eddington--Finkelstein (EF) advanced time.
// Positive particle weights represent invariant rest mass in units where the
// material rest mass and G=c are one. In Schwarzschild coordinates w is
// radial momentum measured by the local static observer, and eta=L^2/2 stores
// specific angular momentum. EF instead stores positive canonical momentum
// P=p_r, whose relation to the observer's momentum includes the metric.
//
// The physical calculation proceeds from particles to stress, radial Einstein
// constraints, metric interpolation, and geodesic updates. The shared headers
// implement those numerical operations. This driver constructs or restores
// their input state, calls them in time order, and writes profiles, histories
// and complete checkpoints. The low-compactness branch also compares particle
// motion and deposited density between Einstein--Vlasov and Vlasov--Poisson.
//
// A fresh EF run requires particles on one ingoing null slice. The coordinate
// mapper obtains them by intersecting Schwarzschild particle trajectories
// with that slice. A Schwarzschild constant-time particle list and an EF
// constant-advanced-time list describe different spacetime surfaces. Their
// identity and coordinate conversion are checked before evolution.
// Schwarzschild evolution stops as its compactness gap approaches zero. EF
// can detect a trapped surface and continue into its interior under the
// explicitly chosen inner-boundary and excision conditions.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ev/weighted_einstein_vlasov_pic.hpp"
#include "ev/weighted_einstein_vlasov_coordinate_map.hpp"
#include "ev/weighted_einstein_vlasov_halo_classifier.hpp"
#include "ev/weighted_einstein_vlasov_null_vacuum_extension.hpp"
#include "ev/weighted_einstein_vlasov_static_shell.hpp"
#include "ev/sha256.hpp"
#include "ev/wpic_task_profile.hpp"
#include "ev/apps/wpic_driver_entry.hpp"

namespace {

// ---- User controls and reported physical diagnostics ---------------------
// Physical initial data, coordinate choices, numerical resolution and output
// controls are kept in one record. Parsing below checks which combinations
// are supported before allocating particles or creating a run directory.
struct RunOptions {
#if defined(VENUS_CPU_EV_WPIC_LOW_PROFILE)
    std::string scenario = "low";
#elif defined(VENUS_CPU_EV_WPIC_BLACK_HOLE_PROFILE)
    std::string scenario = "high";
#else
    std::string scenario = "all";
#endif
    std::string high_initial_condition = "plummer";
    std::string coordinates = "schwarzschild";
    std::string expected_outcome = "black_hole";
    std::string high_inner_boundary = "origin_parity";
    std::string high_radial_grid = "linear";
    std::filesystem::path output_dir =
        "output/weighted_einstein_vlasov_pic";
    std::filesystem::path physical_ic_path;
    // A Schwarzschild donor writes this solved v=const null slice; an EF run
    // consumes it.  It is deliberately distinct from an ordinary restart.
    std::filesystem::path export_ef_null_initial_data_path;
    std::filesystem::path ef_null_initial_data_path;
    double ef_null_donor_duration = -1.0;
    // Optional authenticated map boundary inside the EF evolution domain.
    // The null slice is solved numerically only through this exact mesh edge
    // and continued as analytic Schwarzschild vacuum to high_r_max.
    double ef_map_worldtube_radius = -1.0;
    std::size_t low_markers = 16000;
    std::size_t high_markers = 20000;
    int low_radial_cells = 192;
    int high_radial_cells = 320;
    double low_t_end = 45.0;
    double high_t_end = 8.0;
    double low_dt_max = 0.06;
    double high_dt_max = 0.008;
    double low_cfl = 0.3;
    double high_gap_stop = 0.03;
    double high_r_min = 0.01;
    double high_r_max = 8.0;
    double high_total_mass = 0.42;
    double high_scale_radius = 1.0;
    double high_velocity_factor = 0.20;
    double high_radial_infall = 0.0;
    double high_infall_core_radius = 0.0;
    double high_radial_importance_exponent = 1.0;
    double high_shell_center = 1.80;
    double high_shell_width = 0.18;
    double high_shell_radial_importance_factor = 1.0;
    double high_shell_radial_momentum = -0.62;
    double high_shell_radial_momentum_dispersion = 0.025;
    double high_shell_tangential_velocity_dispersion = 0.012;
    // Continuum parameters for
    // f=C(1-E/E0)_+^k delta(ell-ell0).  The production marker realization is
    // always shifted-stratified.  Its dedicated pair-count interface emits
    // exactly two markers per lattice point. Its pair count specifies the
    // realization independently of the continuum quadrature orders.
    double high_energy_cutoff_amplitude = 0.50;
    double high_energy_cutoff_ell0 = 1.0;
    double high_energy_cutoff_central_y = 0.13;
    double high_energy_cutoff_power = 1.0;
    // A non-negative fixed value replaces the sampled L distribution.  The
    // multiplicative scale is then applied to either the fixed or sampled L.
    double high_specific_angular_momentum = -1.0;
    double high_angular_momentum_scale = 1.0;
    std::uint64_t high_sampling_realization = 0;
    // Negative keeps the immutable base rest weights.  A positive value
    // rescales every invariant marker rest weight by one common factor after
    // the L controls are applied, so the polar-areal initial ADM mass is the
    // requested value.  This defines a clean fixed-M_ADM threshold family.
    double high_target_initial_adm_mass = -1.0;
    double ef_excision_a = -0.01;
    int ef_excision_buffer_cells = 4;
    double ef_excision_horizon_fraction = 0.1;
    bool ef_enable_excision = true;
    double low_total_mass = 2.0e-3;
    double low_velocity_factor = 0.60;
    double low_radial_importance_exponent = 1.0;
    double timestep_omitted_mass_fraction = 1.0e-5;
    int low_output_steps = 20;
    int high_output_steps = 10;
    unsigned int threads = std::max(1U, std::thread::hardware_concurrency());
    bool quick = false;
    bool resume = false;
    bool initialize_only = false;
};

struct RadialDiagnostics {
    double r10 = 0.0;
    double r50 = 0.0;
    double r90 = 0.0;
    double kinetic = 0.0;
    double potential = 0.0;
    double virial_ratio = 0.0;
};

struct SchwarzschildEvolutionStatus {
    bool outer_domain_limited = false;
    double final_time = 0.0;
    double rejected_trial_dt = 0.0;
    std::string termination_detail;
};

struct LowSummary {
    SchwarzschildEvolutionStatus schwarzschild_evolution;
    double initial_compactness = 0.0;
    double maximum_compactness = 0.0;
    double maximum_compactness_time = 0.0;
    double final_compactness = 0.0;
    double minimum_metric_gap = 1.0;
    double maximum_adm_mass_relative_drift = 0.0;
    double initial_virial_ratio = 0.0;
    double final_virial_ratio = 0.0;
    double initial_r50 = 0.0;
    double minimum_r50 = 0.0;
    double final_r50 = 0.0;
    double density_l1_difference = 0.0;
    double particle_rms_radius_difference = 0.0;
    double particle_rms_momentum_difference = 0.0;
    double ev_rest_mass_relative_error = 0.0;
    double vp_rest_mass_relative_error = 0.0;
    bool virialized_halo = false;
    bool agrees_with_vlasov_poisson = false;
};

struct HighSummary {
    SchwarzschildEvolutionStatus schwarzschild_evolution;
    double initial_min_gap = 1.0;
    double final_min_gap = 1.0;
    double initial_min_lapse = 1.0;
    double final_min_lapse = 1.0;
    double final_maximum_log_redshift = 0.0;
    double initial_min_b = 1.0;
    double final_min_b = 1.0;
    double initial_schwarzschild_map_a_linf = 0.0;
    double initial_schwarzschild_map_b_linf = 0.0;
    double initial_schwarzschild_map_m_linf = 0.0;
    double initial_schwarzschild_map_t_rr_relative_l2 = 0.0;
    double initial_schwarzschild_map_s_relative_l2 = 0.0;
    double initial_schwarzschild_map_t_vv_relative_l2 = 0.0;
    double initial_schwarzschild_map_t_vr_relative_l2 = 0.0;
    double initial_schwarzschild_map_t_theta_theta_relative_l2 = 0.0;
    double initial_compactness = 0.0;
    double final_compactness = 0.0;
    // These three masses deliberately have distinct meanings for mapped EF
    // data.  fixed_family_adm_mass is the Schwarzschild t=0 normalization
    // result defining the physical L-family.  null_slice_conserved_adm_mass
    // is the numerical ADM anchor at the end of the reverse donor, while
    // initial_deposited_adm_mass is reconstructed independently from the EF
    // markers on v_bar=0.  Their differences are convergence diagnostics,
    // never reasons to silently redefine the fixed-ADM family.
    double fixed_family_adm_mass = -1.0;
    // Coordinate-independent initially-bound audit for the physical family.
    // Both coordinate runners compute these values from the same normalized
    // Schwarzschild t=0 marker ensemble, never from the later EF null slice.
    double fixed_family_rest_weight = -1.0;
    double fixed_family_binding_energy = 0.0;
    double fixed_family_binding_fraction = -1.0;
    // E=exp(mu)*epsilon would be conserved only in a static spacetime.  These
    // are explicitly instantaneous, frozen-metric t=0 proxy statistics.
    double initial_energy_at_infinity_proxy_minimum = -1.0;
    double initial_energy_at_infinity_proxy_maximum = -1.0;
    double initial_energy_at_infinity_proxy_rest_weighted_mean = -1.0;
    double initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one =
        -1.0;
    double null_slice_conserved_adm_mass = -1.0;
    double initial_deposited_adm_mass = -1.0;
    double initial_adm_mass = 0.0;
    double final_adm_mass = 0.0;
    double initial_rest_weight = 0.0;
    double initial_rest_weight_scale = 1.0;
    double adm_mass_relative_drift = 0.0;
    double rest_mass_relative_error = 0.0;
    double excised_rest_mass = 0.0;
    double escaped_rest_mass = 0.0;
    double apparent_horizon_radius = -1.0;
    double first_trapped_time = -1.0;
    double post_horizon_advanced_time = 0.0;
    // Fail-closed intrinsic EF black-hole evidence.  These are accumulated
    // at every accepted step (not merely at output cadence) and persisted in
    // a checkpoint sidecar so that a resumed evolution cannot silently lose
    // the pre-resume extrema.
    double ef_minimum_p_r_over_evolution = -1.0;
    double ef_minimum_b_over_evolution = -1.0;
    double ef_maximum_excision_radius = 0.0;
    double ef_maximum_excised_rest_weight = 0.0;
    std::uint64_t ef_accepted_step_audit_samples = 0;
    bool ef_post_horizon_audit_complete = false;
    bool ef_finite_fields_throughout = false;
    bool ef_fixed_adm_bookkeeping_throughout = false;
    bool ef_causal_excision_valid_throughout = false;
    bool ef_causal_excision_inside_horizon_seen = false;
    bool ef_excised_interior_rest_weight_seen = false;
    bool ef_requested_end_time_reached = false;
    double ef_final_advanced_time = 0.0;
    double ef_final_active_min_radius = -1.0;
    double ef_first_field_center_radius = 0.0;
    double ef_first_cell_outer_radius = 0.0;
    double ef_active_weight_inside_first_center = 0.0;
    double ef_active_weight_inside_first_cell = 0.0;
    double ef_inner_face_mass = 0.0;
    double ef_inner_face_mb = 0.0;
    double ef_inner_face_b = 1.0;
    bool ef_origin_mass_residual_available = false;
    double ef_origin_mass_residual_over_adm = 0.0;
    double ef_minimum_active_radius_over_first_cell_width = -1.0;
    double ef_density_peak_radius = -1.0;
    std::size_t ef_density_peak_half_maximum_cells = 0;
    std::string ef_spatial_resolution_assessment = "unavailable";
    double ef_last_accepted_dt = 0.0;
    double ef_final_trial_dt = 0.0;
    std::uint64_t ef_active_markers_below_first_field_center = 0;
    std::string ef_termination_reason;
    std::string ef_termination_detail;
    double initial_r50 = 0.0;
    double minimum_r50 = 0.0;
    double final_r50 = 0.0;
    double final_virial_ratio = 0.0;
    double final_anisotropic_jeans_relative_l1 = -1.0;
    double mass_weighted_mean_specific_angular_momentum = 0.0;
    double mass_weighted_rms_specific_angular_momentum = 0.0;
    double maximum_specific_angular_momentum = 0.0;
    bool energy_cutoff_continuum_available = false;
    double energy_cutoff_configured_amplitude = -1.0;
    double energy_cutoff_configured_ell0 = -1.0;
    double energy_cutoff_configured_central_y = -1.0;
    double energy_cutoff_configured_power = -1.0;
    double energy_cutoff_unscaled_adm_mass = -1.0;
    double energy_cutoff_realized_amplitude = -1.0;
    double energy_cutoff_realized_ell0 = -1.0;
    double energy_cutoff_realized_power = -1.0;
    double energy_cutoff_cutoff_energy = -1.0;
    double energy_cutoff_inner_matter_radius = -1.0;
    double energy_cutoff_outer_matter_radius = -1.0;
    double energy_cutoff_continuum_adm_mass = -1.0;
    double energy_cutoff_continuum_rest_weight = -1.0;
    double energy_cutoff_continuum_binding_energy = -1.0;
    double energy_cutoff_continuum_binding_fraction = -1.0;
    double energy_cutoff_central_lapse = -1.0;
    double energy_cutoff_maximum_compactness = -1.0;
    double energy_cutoff_minimum_compactness_gap = -1.0;
    double energy_cutoff_constraint_residual_linf = -1.0;
    double energy_cutoff_jeans_residual_relative_l2 = -1.0;
    double energy_cutoff_sampled_rest_weight = -1.0;
    double energy_cutoff_sampled_rest_weight_relative_error = -1.0;
    double energy_cutoff_sampled_maximum_energy_over_cutoff = -1.0;
    std::size_t energy_cutoff_sampling_pair_count = 0;
    double horizon_time = -1.0;
    std::uint64_t base_ic_fingerprint = 0;
    std::uint64_t realized_ic_fingerprint = 0;
    bool eikonal_mapped_initial_data = false;
    std::string coordinate_map_manifest_sha256;
    double eikonal_boundary_radius = -1.0;
    double eikonal_evolution_outer_radius = -1.0;
    double eikonal_v_offset = 0.0;
    bool eikonal_analytic_vacuum_extension = false;
    int steps = 0;
    bool horizon_formed = false;
    bool evolved_past_horizon = false;
    bool fixed_family_binding_diagnostics_available = false;
    bool fixed_family_positive_binding_energy = false;
    ev::EinsteinVlasovHaloClassification halo_classification;
    bool virialized_halo = false;
    std::string outcome = "undetermined";
};

std::pair<double, double> lapse_diagnostics(
    const ev::Metric& metric) {
    const std::vector<double>& values =
        metric.face_mu.empty() ? metric.mu : metric.face_mu;
    if (values.empty()) {
        throw std::runtime_error(
            "Lapse diagnostics received an empty metric.");
    }
    const double minimum_mu =
        *std::min_element(values.begin(), values.end());
    return {
        std::exp(minimum_mu),
        -minimum_mu,
    };
}

// ---- Command-line interface ----------------------------------------------
// Read values into RunOptions first, then validate their joint physical and
// coordinate assumptions. For example, a null-slice input belongs to EF,
// while a Schwarzschild donor supplies the worldlines used to construct it.
RunOptions parse_options(int argc, char** argv) {
    RunOptions result;
    auto need = [&](int& index) {
        if (index + 1 >= argc) {
            throw std::runtime_error(
                std::string("Missing value after ") + argv[index]);
        }
        return std::string(argv[++index]);
    };
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--scenario") {
            result.scenario = need(i);
        } else if (argument == "--coordinates"
                   || argument == "--spacetime-coordinates") {
            result.coordinates = need(i);
            if (result.coordinates == "ef"
                || result.coordinates == "ingoing_ef"
                || result.coordinates == "ingoing-ef") {
                result.coordinates = "eddington_finkelstein";
            } else if (result.coordinates == "polar_areal"
                       || result.coordinates == "polar-areal") {
                result.coordinates = "schwarzschild";
            }
        } else if (argument == "--expected-outcome") {
            result.expected_outcome = need(i);
        } else if (argument == "--high-initial-condition") {
            result.high_initial_condition = need(i);
        } else if (argument == "--high-inner-boundary") {
            result.high_inner_boundary = need(i);
        } else if (argument == "--high-radial-grid") {
            result.high_radial_grid = need(i);
        } else if (argument == "--output-dir") {
            result.output_dir = need(i);
        } else if (argument == "--physical-ic") {
            result.physical_ic_path = need(i);
        } else if (argument == "--export-ef-null-initial-data") {
            result.export_ef_null_initial_data_path = need(i);
        } else if (argument == "--ef-null-initial-data") {
            result.ef_null_initial_data_path = need(i);
        } else if (argument == "--ef-null-donor-duration") {
            result.ef_null_donor_duration = std::stod(need(i));
        } else if (argument == "--ef-map-worldtube-radius") {
            result.ef_map_worldtube_radius = std::stod(need(i));
        } else if (argument == "--threads") {
            result.threads =
                static_cast<unsigned int>(std::stoul(need(i)));
        } else if (argument == "--low-markers") {
            result.low_markers =
                static_cast<std::size_t>(std::stoull(need(i)));
        } else if (argument == "--high-markers") {
            result.high_markers =
                static_cast<std::size_t>(std::stoull(need(i)));
        } else if (argument == "--low-radial-cells") {
            result.low_radial_cells = std::stoi(need(i));
        } else if (argument == "--high-radial-cells") {
            result.high_radial_cells = std::stoi(need(i));
        } else if (argument == "--low-t-end") {
            result.low_t_end = std::stod(need(i));
        } else if (argument == "--high-t-end") {
            result.high_t_end = std::stod(need(i));
        } else if (argument == "--low-dt-max") {
            result.low_dt_max = std::stod(need(i));
        } else if (argument == "--high-dt-max") {
            result.high_dt_max = std::stod(need(i));
        } else if (argument == "--low-cfl") {
            result.low_cfl = std::stod(need(i));
        } else if (argument == "--high-gap-stop") {
            result.high_gap_stop = std::stod(need(i));
        } else if (argument == "--high-r-min") {
            result.high_r_min = std::stod(need(i));
        } else if (argument == "--high-r-max") {
            result.high_r_max = std::stod(need(i));
        } else if (argument == "--high-total-mass") {
            result.high_total_mass = std::stod(need(i));
        } else if (argument == "--high-scale-radius") {
            result.high_scale_radius = std::stod(need(i));
        } else if (argument == "--high-velocity-factor") {
            result.high_velocity_factor = std::stod(need(i));
        } else if (argument == "--high-radial-infall") {
            result.high_radial_infall = std::stod(need(i));
        } else if (argument == "--high-infall-core-radius") {
            result.high_infall_core_radius = std::stod(need(i));
        } else if (argument == "--high-radial-importance-exponent") {
            result.high_radial_importance_exponent = std::stod(need(i));
        } else if (argument == "--high-shell-center") {
            result.high_shell_center = std::stod(need(i));
        } else if (argument == "--high-shell-width") {
            result.high_shell_width = std::stod(need(i));
        } else if (argument == "--high-shell-radial-importance-factor") {
            result.high_shell_radial_importance_factor =
                std::stod(need(i));
        } else if (argument == "--high-shell-radial-momentum") {
            result.high_shell_radial_momentum = std::stod(need(i));
        } else if (argument
                   == "--high-shell-radial-momentum-dispersion") {
            result.high_shell_radial_momentum_dispersion =
                std::stod(need(i));
        } else if (argument
                   == "--high-shell-tangential-velocity-dispersion") {
            result.high_shell_tangential_velocity_dispersion =
                std::stod(need(i));
        } else if (argument == "--high-energy-cutoff-amplitude") {
            result.high_energy_cutoff_amplitude = std::stod(need(i));
        } else if (argument == "--high-energy-cutoff-ell0") {
            result.high_energy_cutoff_ell0 = std::stod(need(i));
        } else if (argument == "--high-energy-cutoff-central-y") {
            result.high_energy_cutoff_central_y = std::stod(need(i));
        } else if (argument == "--high-energy-cutoff-power") {
            result.high_energy_cutoff_power = std::stod(need(i));
        } else if (argument == "--high-specific-angular-momentum") {
            result.high_specific_angular_momentum = std::stod(need(i));
        } else if (argument == "--high-angular-momentum-scale") {
            result.high_angular_momentum_scale = std::stod(need(i));
        } else if (argument == "--high-sampling-realization") {
            result.high_sampling_realization = std::stoull(need(i));
        } else if (argument == "--high-target-initial-adm-mass") {
            result.high_target_initial_adm_mass = std::stod(need(i));
        } else if (argument == "--ef-evolve-interior") {
            result.ef_enable_excision = false;
        } else if (argument == "--ef-excision-a") {
            result.ef_excision_a = std::stod(need(i));
        } else if (argument == "--ef-excision-buffer-cells") {
            result.ef_excision_buffer_cells = std::stoi(need(i));
        } else if (argument == "--ef-excision-horizon-fraction") {
            result.ef_excision_horizon_fraction = std::stod(need(i));
        } else if (argument == "--low-total-mass") {
            result.low_total_mass = std::stod(need(i));
        } else if (argument == "--low-velocity-factor") {
            result.low_velocity_factor = std::stod(need(i));
        } else if (argument == "--low-radial-importance-exponent") {
            result.low_radial_importance_exponent = std::stod(need(i));
        } else if (argument == "--timestep-omitted-mass-fraction") {
            result.timestep_omitted_mass_fraction =
                std::stod(need(i));
        } else if (argument == "--low-output-steps") {
            result.low_output_steps = std::stoi(need(i));
        } else if (argument == "--high-output-steps") {
            result.high_output_steps = std::stoi(need(i));
        } else if (argument == "--resume") {
            result.resume = true;
        } else if (argument == "--initialize-only") {
            result.initialize_only = true;
        } else if (argument == "--quick") {
            result.quick = true;
            result.low_markers = 8000;
            result.high_markers = 10000;
            result.low_radial_cells = 128;
            result.high_radial_cells = 224;
            result.low_t_end = 32.0;
            result.high_t_end = 7.0;
        } else if (argument == "--help" || argument == "-h") {
#if defined(VENUS_CPU_EV_WPIC_LOW_PROFILE)
            std::cout
                << "Usage: " << argv[0]
                << " [--output-dir PATH] [--threads N] "
                   "[--physical-ic PATH] [--initialize-only] "
                   "[--export-ef-null-initial-data PATH] "
                   "[--ef-null-initial-data PATH] "
                   "[--ef-null-donor-duration T] "
                   "[--ef-map-worldtube-radius R_B] "
                   "[--low-markers N] [--low-radial-cells N] "
                   "[--low-t-end T] [--low-dt-max DT] [--low-cfl C] "
                   "[--low-total-mass M] [--low-velocity-factor F] "
                   "[--low-radial-importance-exponent P] "
                   "[--timestep-omitted-mass-fraction F] "
                   "[--low-output-steps N] [--resume] [--quick]\n\n"
                   "Locked task profile: low-compactness virialized-halo "
                   "Einstein-Vlasov WPIC.  --scenario is unnecessary and, "
                   "if supplied, must be low.\n";
#elif defined(VENUS_CPU_EV_WPIC_BLACK_HOLE_PROFILE)
            std::cout
                << "Usage: " << argv[0]
                << " [--output-dir PATH] [--threads N] "
                   "[--physical-ic PATH] [--initialize-only] "
                   "[--export-ef-null-initial-data PATH] "
                   "[--ef-null-initial-data PATH] "
                   "[--ef-null-donor-duration T] "
                   "[--ef-map-worldtube-radius R_B] "
                   "[--coordinates schwarzschild|eddington_finkelstein] "
                   "[--expected-outcome black_hole|halo|any] "
                   "[--high-markers N] [--high-radial-cells N] "
                   "[--high-t-end T] [--high-dt-max DT] "
                   "[--high-gap-stop G] [--high-r-min R] [--high-r-max R] "
                   "[--high-total-mass M] [--high-initial-condition shell|plummer|energy_cutoff_shell] "
                   "[--high-shell-center R] [--high-shell-width SIGMA] "
                   "[--high-shell-radial-importance-factor F] "
                   "[--high-shell-radial-momentum W] "
                   "[--high-shell-radial-momentum-dispersion SIGMA_W] "
                   "[--high-shell-tangential-velocity-dispersion SIGMA_VT] "
                   "[--high-energy-cutoff-amplitude C] "
                   "[--high-energy-cutoff-ell0 ELL0] "
                   "[--high-energy-cutoff-central-y Y0] "
                   "[--high-energy-cutoff-power K] "
                   "[--high-specific-angular-momentum L] "
                   "[--high-angular-momentum-scale S] "
                   "[--high-sampling-realization UINT64] "
                   "[--high-target-initial-adm-mass M] "
                   "[--timestep-omitted-mass-fraction F] "
                   "[--ef-excision-a A] [--ef-excision-buffer-cells N] "
                   "[--ef-excision-horizon-fraction F] [--ef-evolve-interior] "
                   "[--high-inner-boundary reflecting|origin_parity] "
                   "[--high-radial-grid log|linear] [--high-output-steps N] "
                   "[--resume] [--quick]\n\n"
                   "Locked task profile: high-compactness collapse / "
                   "black-hole-formation Einstein-Vlasov WPIC.  --scenario "
                   "is unnecessary and, if supplied, must be high.\n";
#else
            std::cout
                << "Usage: " << argv[0] << ' '
                << "[--scenario all|low|high] [--output-dir PATH] "
                   "[--physical-ic PATH] [--initialize-only] "
                   "[--export-ef-null-initial-data PATH] "
                   "[--ef-null-initial-data PATH] "
                   "[--ef-null-donor-duration T] "
                   "[--ef-map-worldtube-radius R_B] "
                   "[--coordinates schwarzschild|eddington_finkelstein] "
                   "[--expected-outcome black_hole|halo|any] "
                   "[--high-initial-condition shell|plummer|energy_cutoff_shell] "
                   "[--high-inner-boundary reflecting|origin_parity] "
                   "[--high-radial-grid log|linear] "
                   "[--threads N] [--low-markers N] [--high-markers N] "
                   "[--low-radial-cells N] [--high-radial-cells N] "
                   "[--low-t-end T] [--high-t-end T] "
                   "[--low-dt-max DT] [--high-dt-max DT] "
                   "[--low-cfl C] [--high-gap-stop G] "
                   "[--high-r-min R] [--high-r-max R] "
                   "[--high-total-mass M] [--high-scale-radius A] "
                   "[--high-velocity-factor F] "
                   "[--high-radial-infall W] "
                   "[--high-infall-core-radius R] "
                   "[--high-radial-importance-exponent P] "
                   "[--high-shell-center R] [--high-shell-width SIGMA] "
                   "[--high-shell-radial-importance-factor F] "
                   "[--high-shell-radial-momentum W] "
                   "[--high-shell-radial-momentum-dispersion SIGMA_W] "
                   "[--high-shell-tangential-velocity-dispersion SIGMA_VT] "
                   "[--high-energy-cutoff-amplitude C] "
                   "[--high-energy-cutoff-ell0 ELL0] "
                   "[--high-energy-cutoff-central-y Y0] "
                   "[--high-energy-cutoff-power K] "
                   "[--high-specific-angular-momentum L] "
                   "[--high-angular-momentum-scale S] "
                   "[--high-sampling-realization UINT64] "
                   "[--high-target-initial-adm-mass M] "
                   "[--ef-excision-a A] [--ef-excision-buffer-cells N] "
                   "[--ef-excision-horizon-fraction F] [--ef-evolve-interior] "
                   "[--low-total-mass M] [--low-velocity-factor F] "
                   "[--low-radial-importance-exponent P] "
                   "[--timestep-omitted-mass-fraction F] "
                   "[--low-output-steps N] [--high-output-steps N] "
                   "[--resume] [--quick]\n"
                   "\n"
                   "--resume reads atomic particle checkpoints from "
                   "--output-dir and appends to the existing CSV output. "
                   "All numerical options must match the original run.\n";
#endif
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }
#if defined(VENUS_CPU_EV_WPIC_LOW_PROFILE)
    ev::validate_einstein_vlasov_wpic_task_profile(
        ev::EinsteinVlasovWPICTaskProfile::low_compactness,
        result.scenario);
#elif defined(VENUS_CPU_EV_WPIC_BLACK_HOLE_PROFILE)
    ev::validate_einstein_vlasov_wpic_task_profile(
        ev::EinsteinVlasovWPICTaskProfile::black_hole_formation,
        result.scenario);
#endif
    if (!(result.scenario == "all" || result.scenario == "low"
          || result.scenario == "high")
        || !(result.coordinates == "schwarzschild"
             || result.coordinates == "eddington_finkelstein")
        || (!result.ef_enable_excision
            && result.coordinates != "eddington_finkelstein")
        || !(result.expected_outcome == "black_hole"
             || result.expected_outcome == "halo"
             || result.expected_outcome == "any")
        || !(result.high_initial_condition == "shell"
             || result.high_initial_condition == "plummer"
             || result.high_initial_condition == "energy_cutoff_shell")
        || !(result.high_inner_boundary == "reflecting"
             || result.high_inner_boundary == "origin_parity")
        || !(result.high_radial_grid == "log"
             || result.high_radial_grid == "linear")
        || result.threads == 0 || result.low_markers < 100
        || result.high_markers < 100 || result.low_radial_cells < 8
        || result.high_radial_cells < 8 || !(result.low_t_end > 0.0)
        || !(result.high_t_end > 0.0) || !(result.low_dt_max > 0.0)
        || !(result.high_dt_max > 0.0)
        || !(result.low_cfl > 0.0)
        || !(result.low_cfl <= 1.0)
        || !(result.high_gap_stop > 0.0)
        || !(result.high_gap_stop < 0.2)
        || !(result.high_r_min > 0.0)
        || !(result.high_r_max > result.high_r_min)
        || !(result.high_total_mass > 0.0)
        || !(result.high_total_mass < 1.0)
        || !(result.high_scale_radius > 0.0)
        || !(result.high_velocity_factor >= 0.0)
        || !(result.high_velocity_factor <= 2.0)
        || !(result.high_radial_infall >= 0.0)
        || !(result.high_radial_infall <= 2.0)
        || !(result.high_infall_core_radius >= 0.0)
        || !(result.high_radial_importance_exponent >= 1.0)
        || !(result.high_radial_importance_exponent <= 4.0)
        || !(result.high_shell_center > result.high_r_min)
        || !(result.high_shell_center < result.high_r_max)
        || !(result.high_shell_width > 0.0)
        || !(result.high_shell_width < result.high_r_max)
        || !(result.high_shell_radial_importance_factor >= 1.0)
        || !(result.high_shell_radial_importance_factor <= 4.0)
        || !(result.high_shell_radial_momentum < 0.0)
        || !(result.high_shell_radial_momentum > -3.0)
        || !(result.high_shell_radial_momentum_dispersion >= 0.0)
        || !(result.high_shell_radial_momentum_dispersion <= 1.0)
        || !(result.high_shell_tangential_velocity_dispersion >= 0.0)
        || !(result.high_shell_tangential_velocity_dispersion <= 1.0)
        || !(result.high_energy_cutoff_amplitude > 0.0)
        || !std::isfinite(result.high_energy_cutoff_amplitude)
        || !(result.high_energy_cutoff_ell0 > 0.0)
        || !std::isfinite(result.high_energy_cutoff_ell0)
        || !(result.high_energy_cutoff_central_y > 0.0)
        || !std::isfinite(result.high_energy_cutoff_central_y)
        || !(result.high_energy_cutoff_power >= 1.0)
        || !(result.high_energy_cutoff_power <= 8.0)
        || !std::isfinite(result.high_energy_cutoff_power)
        || !(result.high_specific_angular_momentum >= -1.0)
        || !std::isfinite(result.high_specific_angular_momentum)
        || !(result.high_angular_momentum_scale >= 0.0)
        || !std::isfinite(result.high_angular_momentum_scale)
        || !std::isfinite(result.high_target_initial_adm_mass)
        || !(result.high_target_initial_adm_mass == -1.0
             || result.high_target_initial_adm_mass > 0.0)
        || !(result.ef_excision_a < 0.0)
        || !std::isfinite(result.ef_excision_a)
        || result.ef_excision_buffer_cells < 1
        || !(result.ef_excision_horizon_fraction > 0.0)
        || !(result.ef_excision_horizon_fraction <= 1.0)
        || !std::isfinite(result.ef_null_donor_duration)
        || !(result.ef_null_donor_duration == -1.0
             || result.ef_null_donor_duration > 0.0)
        || !std::isfinite(result.ef_map_worldtube_radius)
        || !(result.ef_map_worldtube_radius == -1.0
             || (result.ef_map_worldtube_radius > result.high_r_min
                 && result.ef_map_worldtube_radius < result.high_r_max))
        || !(result.low_total_mass > 0.0)
        || !(result.low_total_mass < 0.5)
        || !(result.low_velocity_factor >= 0.0)
        || !(result.low_velocity_factor <= 2.0)
        || !(result.low_radial_importance_exponent >= 1.0)
        || !(result.low_radial_importance_exponent <= 4.0)
        || !(result.timestep_omitted_mass_fraction >= 0.0)
        || !(result.timestep_omitted_mass_fraction < 0.01)
        || result.low_output_steps < 1
        || result.high_output_steps < 1) {
        throw std::runtime_error("Invalid Einstein--Vlasov WPIC options.");
    }
    if (result.coordinates == "eddington_finkelstein"
        && result.scenario != "high") {
        throw std::runtime_error(
            "The WPIC Eddington--Finkelstein module is currently exposed "
            "only by --scenario high; the low EV/VP comparison remains "
            "Schwarzschild/polar-areal.");
    }
    if (result.coordinates == "eddington_finkelstein"
        && (result.high_inner_boundary != "origin_parity"
            || result.high_radial_grid != "linear")) {
        throw std::runtime_error(
            "The ingoing Eddington--Finkelstein shell calculation requires "
            "--high-inner-boundary origin_parity and --high-radial-grid "
            "linear; the center is avoided by shell support and is excised "
            "only after a trapped region forms.");
    }
    // The continuum sampler emits equal-weight (+w,-w) pairs. An even count
    // preserves zero mean radial momentum before any prescribed inward kick.
    if (result.high_initial_condition == "energy_cutoff_shell") {
        if (result.high_markers % 2 != 0) {
            throw std::runtime_error(
                "--high-markers for energy_cutoff_shell must be even so the "
                "shifted-stratified sampler emits exactly the requested "
                "number of adjacent (+w,-w) markers.");
        }
        if (result.high_specific_angular_momentum >= 0.0) {
            throw std::runtime_error(
                "energy_cutoff_shell obtains its baseline ell0 from "
                "--high-energy-cutoff-ell0; do not replace it with "
                "--high-specific-angular-momentum.  Use "
                "--high-angular-momentum-scale for the clean threshold "
                "family.");
        }
        if (result.resume && result.physical_ic_path.empty()) {
            throw std::runtime_error(
                "An energy_cutoff_shell resume requires --physical-ic so "
                "the configured continuum, realization, and immutable "
                "invariant weights can be authenticated.");
        }
    }
    if (!result.export_ef_null_initial_data_path.empty()
        && (!result.ef_null_initial_data_path.empty()
            || result.coordinates != "schwarzschild"
            || result.scenario != "high" || result.resume)) {
        throw std::runtime_error(
            "--export-ef-null-initial-data is a fresh Schwarzschild high-"
            "compactness donor operation and cannot be combined with "
            "--ef-null-initial-data or --resume.");
    }
    if (result.ef_map_worldtube_radius > 0.0
        && (result.export_ef_null_initial_data_path.empty()
            || result.ef_null_donor_duration <= 0.0
            || result.high_inner_boundary != "origin_parity"
            || result.high_radial_grid != "linear")) {
        throw std::runtime_error(
            "--ef-map-worldtube-radius requires a fresh solved-null export, "
            "an explicit --ef-null-donor-duration, origin_parity, and a "
            "linear radial grid.");
    }
    if (!result.ef_null_initial_data_path.empty()
        && (result.coordinates != "eddington_finkelstein"
            || result.scenario != "high")) {
        throw std::runtime_error(
            "--ef-null-initial-data is accepted only by the high-"
            "compactness Eddington--Finkelstein module.");
    }
    if (result.coordinates == "eddington_finkelstein" && !result.resume
        && result.ef_null_initial_data_path.empty()) {
        throw std::runtime_error(
            "A fresh EF evolution requires --ef-null-initial-data from a "
            "solved Schwarzschild eikonal donor.  Equal coordinate labels "
            "on t=const and v=const are not physical initial data.");
    }
    if (result.initialize_only
        && (result.scenario != "high" || result.resume
            || result.physical_ic_path.empty())) {
        throw std::runtime_error(
            "--initialize-only requires --scenario high, a "
            "--physical-ic path, and a fresh (non-resume) run.");
    }
    return result;
}

// ---- Shared radial mesh and deterministic initial-condition samplers -----
// Use a spherical radial mesh for deposition and metric reconstruction.
// The marker drivers carry the momentum coordinates directly, so the small
// auxiliary momentum-grid dimensions here do not set particle resolution.
ev::Params base_parameters(int radial_cells, double r_min, double r_max) {
    ev::Params p;
    p.physics_model = "einstein_vlasov";
    p.vp_universe = "static";
    p.bp_collision_operator = "none";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "uniform";
    p.vp_interpolation = "hamiltonian_sldg";
    p.vp_poisson_reconstruction = "p2_line_mass";
    p.integrator = "semi_lagrangian_kick_drift_kick";
    p.Nr = radial_cells;
    p.Nw = 4;
    p.NL = 2;
    p.r_min = r_min;
    p.r_max = r_max;
    p.radial_grid = "log";
    p.w_max = 2.0;
    p.L_max = 4.0;
    p.inner_boundary = "reflecting";
    p.outer_boundary = "outflow_vacuum";
    p.metric_floor = 1.0e-10;
    p.gap_stop = 0.03;
    p.vp_timestep_omitted_mass_fraction = 1.0e-5;
    return p;
}

ev::WeightedPICMarker make_marker(
    double mass,
    double radius,
    double radial_momentum,
    double eta,
    std::uint64_t id) {
    ev::WeightedPICMarker result;
    result.mass = mass;
    result.r = radius;
    result.w = radial_momentum;
    result.eta = eta;
    result.id = id;
    result.source_cell = id;
    return result;
}

double uniform(std::uint64_t seed, std::uint64_t counter, std::uint64_t dim) {
    return ev::weighted_pic_counter_uniform(seed, counter, dim);
}

// Reverse the digits of the integer into fractional places.  In base two,
// indices 1,2,3 give 1/2,1/4,3/4.  These distributed quantiles sample a
// velocity distribution regularly before any optional realization rotation.
double radical_inverse(std::uint64_t index, std::uint64_t base) {
    double result = 0.0;
    double place = 1.0 / static_cast<double>(base);
    while (index != 0) {
        const std::uint64_t digit = index % base;
        result += static_cast<double>(digit) * place;
        index /= base;
        place /= static_cast<double>(base);
    }
    return result;
}

// Convert a probability into a Gaussian deviate by bisection of its CDF.
// Endpoint clipping keeps the inverse finite at representable probabilities.
double inverse_standard_normal_cdf(double probability) {
    probability = std::max(
        std::numeric_limits<double>::epsilon(),
        std::min(
            probability,
            1.0 - std::numeric_limits<double>::epsilon()));
    double lower = -8.2;
    double upper = 8.2;
    constexpr double inverse_sqrt_two =
        0.707106781186547524400844362104849039;
    for (int iteration = 0; iteration < 56; ++iteration) {
        const double midpoint = 0.5 * (lower + upper);
        const double cumulative =
            0.5 * std::erfc(-midpoint * inverse_sqrt_two);
        if (cumulative < probability) {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    return 0.5 * (lower + upper);
}

double plummer_enclosed_fraction(double radius, double scale_radius) {
    const double radius_squared = radius * radius;
    return radius * radius_squared
        / std::pow(
            radius_squared + scale_radius * scale_radius, 1.5);
}

double inverse_plummer_radius(double enclosed_fraction, double scale_radius) {
    return scale_radius
        / std::sqrt(
            std::pow(enclosed_fraction, -2.0 / 3.0) - 1.0);
}

// Sample the truncated Plummer enclosed-mass profile.  An importance map
// u -> u^p places extra markers in the center.  Its derivative p*u^(p-1)
// supplies their unequal weights, which are normalized to the total mass.
// The velocity factor then sets how cold the sampled Plummer velocities are.
std::vector<ev::WeightedPICMarker> make_cold_plummer_markers(
    std::size_t count,
    double total_mass,
    double scale_radius,
    double r_min,
    double r_max,
    double velocity_factor,
    double radial_importance_exponent,
    std::uint64_t sampling_realization) {
    const double enclosed_min =
        plummer_enclosed_fraction(r_min, scale_radius);
    const double enclosed_max =
        plummer_enclosed_fraction(r_max, scale_radius);
    const double velocity_scale = std::sqrt(total_mass / scale_radius);
    std::vector<ev::WeightedPICMarker> result;
    result.reserve(count);
    const std::uint64_t seed = UINT64_C(0x3141592653589793)
        ^ (sampling_realization * UINT64_C(0x9e3779b97f4a7c15));
    const double stratum_offset = sampling_realization == 0
        ? 0.5 : uniform(seed, sampling_realization, 11);
    std::vector<double> raw_marker_weight(count, 0.0);
    long double raw_weight_sum = 0.0L;
    for (std::size_t q = 0; q < count; ++q) {
        const double sampling_coordinate =
            (static_cast<double>(q) + stratum_offset)
                / static_cast<double>(count);
        raw_marker_weight[q] =
            radial_importance_exponent
            * std::pow(
                sampling_coordinate,
                radial_importance_exponent - 1.0);
        raw_weight_sum += raw_marker_weight[q];
    }
    for (std::size_t q = 0; q < count; ++q) {
        const double sampling_coordinate =
            (static_cast<double>(q) + stratum_offset)
                / static_cast<double>(count);
        const double mass_coordinate = enclosed_min
            + std::pow(
                sampling_coordinate,
                radial_importance_exponent)
                * (enclosed_max - enclosed_min);
        const double radius =
            inverse_plummer_radius(mass_coordinate, scale_radius);
        // Rejection-sample x=v/v_escape with density x^2*(1-x^2)^(7/2).
        // The x^2 factor is the velocity-space volume element.  A separate
        // uniform cosine below supplies the isotropic direction.
        double speed_fraction = 0.0;
        bool accepted = false;
        for (std::uint64_t attempt = 0; attempt < 100000; ++attempt) {
            const std::uint64_t counter =
                static_cast<std::uint64_t>(q) * UINT64_C(100003)
                + attempt;
            const double trial = uniform(seed, counter, 0);
            const double ordinate = 0.1 * uniform(seed, counter, 1);
            const double density =
                trial * trial
                * std::pow(std::max(0.0, 1.0 - trial * trial), 3.5);
            if (ordinate <= density) {
                speed_fraction = trial;
                accepted = true;
                break;
            }
        }
        if (!accepted) {
            throw std::runtime_error(
                "Plummer velocity rejection sampler did not converge.");
        }
        const double escape_speed =
            std::sqrt(2.0) * velocity_scale
            * std::pow(
                1.0 + radius * radius
                    / (scale_radius * scale_radius),
                -0.25);
        const double speed =
            velocity_factor * speed_fraction * escape_speed;
        const double cosine =
            2.0 * uniform(seed, static_cast<std::uint64_t>(q), 2) - 1.0;
        const double radial_momentum = speed * cosine;
        const double tangential_speed =
            speed * std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
        result.push_back(make_marker(
            total_mass * raw_marker_weight[q]
                / static_cast<double>(raw_weight_sum),
            radius,
            radial_momentum,
            0.5 * radius * radius * tangential_speed * tangential_speed,
            static_cast<std::uint64_t>(q + 1)));
    }
    return result;
}

std::vector<ev::WeightedPICMarker> make_infalling_shell_markers(
    std::size_t count,
    double rest_mass,
    double inner_radius,
    double outer_radius,
    double shell_center,
    double shell_width,
    double radial_importance_factor,
    double mean_radial_momentum,
    double radial_momentum_dispersion,
    double tangential_velocity_dispersion,
    std::uint64_t sampling_realization) {
    if (count == 0 || !(rest_mass > 0.0)
        || !(outer_radius > inner_radius)
        || !(shell_center > inner_radius)
        || !(shell_center < outer_radius)
        || !(shell_width > 0.0)
        || !(radial_importance_factor >= 1.0)
        || !(mean_radial_momentum < 0.0)
        || !(radial_momentum_dispersion >= 0.0)
        || !(tangential_velocity_dispersion >= 0.0)) {
        throw std::runtime_error(
            "Invalid Gaussian infalling-shell parameters.");
    }

    // Sample the spherical mass density
    //
    //   rho(r) proportional to exp[-(r-r_shell)^2/(2 sigma_r^2)]
    //
    // rather than sampling a Gaussian line density.  Thus the target radial
    // probability is proportional to r^2 rho(r).  The inverse-CDF proposal may
    // use a broader Gaussian, with exact target/proposal marker weights, to
    // resolve the low-density material on both sides of the eventual horizon.
    // Deterministic radial quantiles remove Monte Carlo radial shot noise,
    // while the momenta retain a finite smooth dispersion.
    constexpr std::size_t cdf_intervals = 65536;
    std::vector<double> cdf_radius(cdf_intervals + 1, 0.0);
    std::vector<double> cdf(cdf_intervals + 1, 0.0);
    const double proposal_width =
        radial_importance_factor * shell_width;
    auto proposal_radial_probability_density = [&](double radius) {
        const double offset = (radius - shell_center) / proposal_width;
        return radius * radius * std::exp(-0.5 * offset * offset);
    };
    const double dr =
        (outer_radius - inner_radius)
        / static_cast<double>(cdf_intervals);
    cdf_radius[0] = inner_radius;
    double previous_density =
        proposal_radial_probability_density(inner_radius);
    for (std::size_t i = 1; i <= cdf_intervals; ++i) {
        const double radius =
            inner_radius + static_cast<double>(i) * dr;
        const double density =
            proposal_radial_probability_density(radius);
        cdf_radius[i] = radius;
        cdf[i] = cdf[i - 1]
            + 0.5 * dr * (previous_density + density);
        previous_density = density;
    }
    if (!(cdf.back() > 0.0) || !std::isfinite(cdf.back())) {
        throw std::runtime_error(
            "Gaussian infalling-shell CDF has zero or invalid mass.");
    }

    std::vector<ev::WeightedPICMarker> result;
    result.reserve(count);
    const std::uint64_t realization_seed =
        UINT64_C(0x6a09e667f3bcc909)
        ^ (sampling_realization * UINT64_C(0x9e3779b97f4a7c15));
    const double radial_stratum_offset = sampling_realization == 0
        ? 0.5 : uniform(realization_seed, sampling_realization, 17);
    const double normal_rotation = sampling_realization == 0
        ? 0.0 : uniform(realization_seed, sampling_realization, 18);
    const double tangential_rotation = sampling_realization == 0
        ? 0.0 : uniform(realization_seed, sampling_realization, 19);
    double raw_weight_sum = 0.0;
    for (std::size_t q = 0; q < count; ++q) {
        const double mass_quantile =
            (static_cast<double>(q) + radial_stratum_offset)
                / static_cast<double>(count);
        const double target = mass_quantile * cdf.back();
        const auto upper = std::lower_bound(cdf.begin(), cdf.end(), target);
        std::size_t upper_index =
            static_cast<std::size_t>(upper - cdf.begin());
        upper_index = std::max<std::size_t>(
            1, std::min(upper_index, cdf_intervals));
        const std::size_t lower_index = upper_index - 1;
        const double denominator =
            cdf[upper_index] - cdf[lower_index];
        const double fraction = denominator > 0.0
            ? (target - cdf[lower_index]) / denominator
            : 0.0;
        const double radius =
            cdf_radius[lower_index]
            + fraction
                * (cdf_radius[upper_index] - cdf_radius[lower_index]);
        const double target_offset =
            (radius - shell_center) / shell_width;
        const double proposal_offset =
            (radius - shell_center) / proposal_width;
        const double raw_marker_weight = std::exp(
            -0.5 * target_offset * target_offset
            + 0.5 * proposal_offset * proposal_offset);
        raw_weight_sum += raw_marker_weight;

        // Adjacent-radius antithetic pairs cancel the local radial-momentum
        // perturbation to leading order.  Radical-inverse quantiles replace
        // random velocity draws, so every local block covers the Gaussian and
        // Rayleigh distributions without Poisson clumping.
        const std::uint64_t pair =
            static_cast<std::uint64_t>(q / 2);
        const double rotated_normal_quantile = std::fmod(
            radical_inverse(pair + 1, 2) + normal_rotation, 1.0);
        double standard_normal = inverse_standard_normal_cdf(
            0.5 + 0.5 * rotated_normal_quantile);
        if (q % 2 != 0) {
            standard_normal = -standard_normal;
        }
        const double radial_momentum =
            mean_radial_momentum
            + radial_momentum_dispersion * standard_normal;
        const double tangential_uniform = std::min(
            std::fmod(
                radical_inverse(pair + 1, 3) + tangential_rotation,
                1.0),
            1.0 - std::numeric_limits<double>::epsilon());
        const double tangential_speed =
            tangential_velocity_dispersion
            * std::sqrt(-2.0 * std::log(1.0 - tangential_uniform));
        result.push_back(make_marker(
            raw_marker_weight,
            radius,
            radial_momentum,
            0.5 * radius * radius
                * tangential_speed * tangential_speed,
            static_cast<std::uint64_t>(q + 1)));
    }
    if (!(raw_weight_sum > 0.0) || !std::isfinite(raw_weight_sum)) {
        throw std::runtime_error(
            "Gaussian infalling-shell importance weights are invalid.");
    }
    for (ev::WeightedPICMarker& marker : result) {
        marker.mass *= rest_mass / raw_weight_sum;
    }
    return result;
}

// A prescribed L replaces eta=L^2/2.  A subsequent scale of L multiplies
// eta by its square.  These controls leave positions and radial momenta
// unchanged, and the optional ADM normalization below adjusts rest weights.
void apply_high_angular_momentum_controls(
    std::vector<ev::WeightedPICMarker>& markers,
    const RunOptions& options) {
    const double scale_squared =
        options.high_angular_momentum_scale
        * options.high_angular_momentum_scale;
    for (ev::WeightedPICMarker& marker : markers) {
        if (options.high_specific_angular_momentum >= 0.0) {
            marker.eta = 0.5
                * options.high_specific_angular_momentum
                * options.high_specific_angular_momentum;
        }
        marker.eta *= scale_squared;
    }
}

struct InitialADMNormalization {
    double rest_weight_scale = 1.0;
    double achieved_adm_mass = -1.0;
};

// Define a one-parameter angular-momentum experiment without inadvertently
// adding total gravitating mass as L is raised.  The phase-space locations,
// radial momenta, eta values, marker IDs, and relative rest weights are held
// fixed; only one common positive multiplier alpha is applied to every
// invariant rest weight.  The polar-areal radial constraint is then solved
// at each trial alpha.  Its ADM mass is monotone on the exterior branch, so a
// bracketed bisection is deterministic and fails closed if the requested mass
// cannot be reached before that branch ends.
InitialADMNormalization normalize_high_initial_adm_mass(
    std::vector<ev::WeightedPICMarker>& markers,
    const RunOptions& options,
    const ev::WeightedPICFieldGrid& grid) {
    InitialADMNormalization result;
    if (options.high_target_initial_adm_mass < 0.0) {
        return result;
    }
    if (markers.empty()) {
        throw std::runtime_error(
            "Fixed-ADM normalization received no high-compactness markers.");
    }
    std::vector<double> unscaled_weight;
    unscaled_weight.reserve(markers.size());
    for (const auto& marker : markers) {
        if (!(marker.mass > 0.0) || !std::isfinite(marker.mass)) {
            throw std::runtime_error(
                "Fixed-ADM normalization requires positive finite weights.");
        }
        unscaled_weight.push_back(marker.mass);
    }
    auto set_scale = [&](double scale) {
        if (!(scale > 0.0) || !std::isfinite(scale)) {
            throw std::runtime_error(
                "Fixed-ADM normalization generated an invalid scale.");
        }
        for (std::size_t q = 0; q < markers.size(); ++q) {
            markers[q].mass = scale * unscaled_weight[q];
        }
    };
    auto adm_at = [&](double scale, bool& exterior) {
        set_scale(scale);
        try {
            const auto field =
                ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
                    markers,
                    grid,
                    ev::WeightedPICParticleShape::tsc,
                    options.threads);
            exterior = true;
            return field.adm_mass;
        } catch (const ev::WeightedEinsteinVlasovPICPolarArealHorizon&) {
            exterior = false;
            return std::numeric_limits<double>::infinity();
        }
    };

    const double target = options.high_target_initial_adm_mass;
    double lower_scale = 0.0;
    double lower_mass = 0.0;
    double upper_scale = 1.0;
    bool upper_exterior = true;
    double upper_mass = adm_at(upper_scale, upper_exterior);
    for (int expansion = 0;
         upper_exterior && upper_mass < target && expansion < 60;
         ++expansion) {
        lower_scale = upper_scale;
        lower_mass = upper_mass;
        upper_scale *= 2.0;
        upper_mass = adm_at(upper_scale, upper_exterior);
    }
    if (upper_exterior && upper_mass < target) {
        throw std::runtime_error(
            "Could not bracket the requested fixed initial ADM mass.");
    }

    // A horizon at the upper trial is a valid bracket endpoint, but the final
    // accepted state must itself remain in the polar exterior.
    for (int iteration = 0; iteration < 64; ++iteration) {
        const double middle_scale = 0.5 * (lower_scale + upper_scale);
        bool middle_exterior = true;
        const double middle_mass = adm_at(middle_scale, middle_exterior);
        if (middle_exterior && middle_mass < target) {
            lower_scale = middle_scale;
            lower_mass = middle_mass;
        } else {
            upper_scale = middle_scale;
            upper_mass = middle_mass;
            upper_exterior = middle_exterior;
        }
    }

    // Choose the closest exterior endpoint.  The lower endpoint is always
    // exterior; the upper one is eligible only if its solve succeeded.
    double selected_scale = lower_scale;
    double selected_mass = lower_mass;
    if (upper_exterior
        && std::abs(upper_mass - target) < std::abs(lower_mass - target)) {
        selected_scale = upper_scale;
        selected_mass = upper_mass;
    }
    set_scale(selected_scale);
    const double relative_error = std::abs(selected_mass - target)
        / std::max(target, 1.0e-300);
    if (!(selected_scale > 0.0) || !std::isfinite(selected_mass)
        || relative_error > 5.0e-11) {
        throw std::runtime_error(
            "Fixed-ADM normalization did not reach the requested mass on "
            "the exterior Schwarzschild branch.");
    }
    result.rest_weight_scale = selected_scale;
    result.achieved_adm_mass = selected_mass;
    return result;
}

struct AngularMomentumDiagnostics {
    double mean = 0.0;
    double rms = 0.0;
    double maximum = 0.0;
};

// eta stores L^2/2. Recover L for each marker, then form rest-weighted
// first and second moments. Angular momentum itself is specific to unit mass.
template <class Marker>
AngularMomentumDiagnostics angular_momentum_diagnostics(
    const std::vector<Marker>& markers) {
    long double total_mass = 0.0L;
    long double first_moment = 0.0L;
    long double second_moment = 0.0L;
    double maximum = 0.0;
    for (const Marker& marker : markers) {
        const double angular_momentum =
            std::sqrt(std::max(0.0, 2.0 * marker.eta));
        total_mass += marker.mass;
        first_moment += marker.mass * angular_momentum;
        second_moment += marker.mass
            * angular_momentum * angular_momentum;
        maximum = std::max(maximum, angular_momentum);
    }
    if (!(total_mass > 0.0L)) {
        throw std::runtime_error(
            "Angular-momentum diagnostics received zero marker mass.");
    }
    return {
        static_cast<double>(first_moment / total_mass),
        std::sqrt(static_cast<double>(second_moment / total_mass)),
        maximum,
    };
}

AngularMomentumDiagnostics angular_momentum_diagnostics(
    const std::vector<ev::eddington_finkelstein::Marker>& markers) {
    long double total_rest_weight = 0.0L;
    long double first_moment = 0.0L;
    long double second_moment = 0.0L;
    double maximum = 0.0;
    for (const auto& marker : markers) {
        const double angular_momentum =
            std::sqrt(std::max(0.0, 2.0 * marker.eta));
        total_rest_weight += marker.rest_weight;
        first_moment += marker.rest_weight * angular_momentum;
        second_moment += marker.rest_weight
            * angular_momentum * angular_momentum;
        maximum = std::max(maximum, angular_momentum);
    }
    if (!(total_rest_weight > 0.0L)) {
        throw std::runtime_error(
            "EF angular-momentum diagnostics received zero rest weight.");
    }
    return {
        static_cast<double>(first_moment / total_rest_weight),
        std::sqrt(static_cast<double>(
            second_moment / total_rest_weight)),
        maximum,
    };
}

struct FixedFamilyBoundDiagnostics {
    double rest_weight = 0.0;
    double adm_mass = 0.0;
    double binding_energy = 0.0;
    double binding_fraction = 0.0;
    double energy_proxy_minimum = 0.0;
    double energy_proxy_maximum = 0.0;
    double energy_proxy_rest_weighted_mean = 0.0;
    double energy_proxy_rest_weight_fraction_ge_one = 0.0;
};

// Evaluate the global binding diagnostic and the optional per-marker proxy on
// the normalized physical Schwarzschild t=0 family.  M0-M_ADM is a global,
// asymptotically normalized statement.  In contrast,
// E_proxy=exp(mu)*sqrt(1+w^2+2 eta/r^2) is a conserved particle energy only
// when the metric is static; for dynamical initial data it is reported solely
// as an instantaneous frozen-metric diagnostic.
FixedFamilyBoundDiagnostics fixed_family_bound_diagnostics(
    const std::vector<ev::WeightedPICMarker>& markers,
    const ev::WeightedPICFieldGrid& grid,
    const ev::WeightedEinsteinVlasovPICField& field) {
    if (markers.empty() || !(field.adm_mass > 0.0)
        || !std::isfinite(field.adm_mass)) {
        throw std::runtime_error(
            "Fixed-family binding diagnostics require a nonempty positive-ADM "
            "Schwarzschild t=0 state.");
    }
    long double rest_weight = 0.0L;
    long double weighted_energy = 0.0L;
    long double proxy_unbound_rest_weight = 0.0L;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = 0.0;
    for (const auto& marker : markers) {
        if (marker.state != ev::WeightedPICMarkerState::active
            || !(marker.mass > 0.0) || !std::isfinite(marker.mass)
            || !(marker.r > 0.0) || !std::isfinite(marker.r)
            || !std::isfinite(marker.w)
            || !(marker.eta >= 0.0) || !std::isfinite(marker.eta)) {
            throw std::runtime_error(
                "Fixed-family binding diagnostics require finite active "
                "positive-weight Schwarzschild t=0 markers.");
        }
        const ev::MetricSample metric =
            ev::weighted_einstein_vlasov_pic_detail::sample_metric_on_grid(
                marker.r, grid, field.metric);
        const long double radius = marker.r;
        const long double epsilon = std::sqrt(
            1.0L
            + static_cast<long double>(marker.w) * marker.w
            + 2.0L * marker.eta / (radius * radius));
        const double energy_proxy = static_cast<double>(
            std::exp(static_cast<long double>(metric.mu)) * epsilon);
        if (!(energy_proxy > 0.0) || !std::isfinite(energy_proxy)) {
            throw std::runtime_error(
                "Initial energy-at-infinity proxy is nonfinite or "
                "nonpositive.");
        }
        const long double weight = marker.mass;
        rest_weight += weight;
        weighted_energy += weight * energy_proxy;
        if (energy_proxy >= 1.0) {
            proxy_unbound_rest_weight += weight;
        }
        minimum = std::min(minimum, energy_proxy);
        maximum = std::max(maximum, energy_proxy);
    }
    if (!(rest_weight > 0.0L) || !std::isfinite(minimum)
        || !(maximum >= minimum)) {
        throw std::runtime_error(
            "Fixed-family binding diagnostics received invalid totals.");
    }
    FixedFamilyBoundDiagnostics result;
    result.rest_weight = static_cast<double>(rest_weight);
    result.adm_mass = field.adm_mass;
    result.binding_energy = result.rest_weight - result.adm_mass;
    result.binding_fraction = result.binding_energy / result.rest_weight;
    result.energy_proxy_minimum = minimum;
    result.energy_proxy_maximum = maximum;
    result.energy_proxy_rest_weighted_mean = static_cast<double>(
        weighted_energy / rest_weight);
    result.energy_proxy_rest_weight_fraction_ge_one = static_cast<double>(
        proxy_unbound_rest_weight / rest_weight);
    return result;
}

void set_fixed_family_bound_diagnostics(
    HighSummary& summary,
    const FixedFamilyBoundDiagnostics& value) {
    summary.fixed_family_rest_weight = value.rest_weight;
    summary.fixed_family_adm_mass = value.adm_mass;
    summary.fixed_family_binding_energy = value.binding_energy;
    summary.fixed_family_binding_fraction = value.binding_fraction;
    summary.initial_energy_at_infinity_proxy_minimum =
        value.energy_proxy_minimum;
    summary.initial_energy_at_infinity_proxy_maximum =
        value.energy_proxy_maximum;
    summary.initial_energy_at_infinity_proxy_rest_weighted_mean =
        value.energy_proxy_rest_weighted_mean;
    summary.initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one =
        value.energy_proxy_rest_weight_fraction_ge_one;
    summary.fixed_family_binding_diagnostics_available = true;
    summary.fixed_family_positive_binding_energy =
        value.binding_fraction > 0.0;
}

void validate_energy_cutoff_fixed_family_binding(
    const RunOptions& options,
    const FixedFamilyBoundDiagnostics& value) {
    if (options.high_initial_condition == "energy_cutoff_shell"
        && (!(value.binding_energy > 0.0)
            || !(value.binding_fraction > 0.0)
            || !std::isfinite(value.binding_fraction))) {
        throw std::runtime_error(
            "The realized energy_cutoff_shell WPIC family is not initially "
            "bound: its invariant rest weight must exceed its polar-areal "
            "ADM mass.");
    }
}

// Hash the ordered binary values of the base particles so an angular-
// momentum scan or restart can identify its exact initial realization.
// SHA-256 file checks are handled separately when authenticating artifacts.
std::uint64_t physical_ic_fingerprint(
    const std::vector<ev::WeightedPICMarker>& markers) {
    constexpr std::uint64_t offset = UINT64_C(1469598103934665603);
    constexpr std::uint64_t prime = UINT64_C(1099511628211);
    std::uint64_t hash = offset;
    auto mix_u64 = [&](std::uint64_t value) {
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= (value >> (8 * byte)) & UINT64_C(0xff);
            hash *= prime;
        }
    };
    auto mix_double = [&](double value) {
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        mix_u64(bits);
    };
    mix_u64(static_cast<std::uint64_t>(markers.size()));
    for (const ev::WeightedPICMarker& marker : markers) {
        mix_double(marker.mass);
        mix_double(marker.r);
        mix_double(marker.w);
        mix_double(marker.eta);
        mix_u64(marker.id);
        mix_u64(marker.source_cell);
    }
    return hash;
}

struct ImmutableMarkerAttributes {
    double rest_weight;
    double eta;
    std::uint64_t source_id;
};

ImmutableMarkerAttributes immutable_marker_attributes(
    const ev::WeightedPICMarker& marker) {
    return {marker.mass, marker.eta, marker.source_cell};
}

ImmutableMarkerAttributes immutable_marker_attributes(
    const ev::eddington_finkelstein::Marker& marker) {
    return {marker.rest_weight, marker.eta, marker.source_id};
}

// Characteristic transport changes position and radial momentum. Invariant
// rest weights, squared angular momenta and marker identities remain fixed.
// Bind a restart to its configured initial realization before opening output.
template <class CheckpointMarker, class InitialMarker>
void validate_restart_initial_marker_invariants(
    const std::vector<CheckpointMarker>& checkpoint,
    const std::vector<InitialMarker>& initial) {
    if (checkpoint.size() != initial.size()) {
        throw std::runtime_error("Restart initial-marker count mismatch.");
    }
    std::unordered_map<std::uint64_t, ImmutableMarkerAttributes> expected;
    expected.reserve(initial.size());
    for (const auto& marker : initial) {
        if (!expected.emplace(marker.id, immutable_marker_attributes(marker)).second) {
            throw std::runtime_error("Duplicate marker ID in initial realization.");
        }
    }
    for (const auto& marker : checkpoint) {
        const auto found = expected.find(marker.id);
        const auto actual = immutable_marker_attributes(marker);
        if (found == expected.end()
            || actual.rest_weight != found->second.rest_weight
            || actual.eta != found->second.eta
            || actual.source_id != found->second.source_id) {
            throw std::runtime_error(
                "Restart immutable marker attributes differ from the configured "
                "physical/null initial data (ID, source ID, rest weight or eta).");
        }
        // Erasing also detects a duplicate checkpoint ID.
        expected.erase(found);
    }
}

template <class T>
void write_physical_ic_scalar(std::ostream& output, const T& value) {
    output.write(
        reinterpret_cast<const char*>(&value),
        static_cast<std::streamsize>(sizeof(T)));
    if (!output) {
        throw std::runtime_error("Failed while writing a physical IC file.");
    }
}

template <class T>
T read_physical_ic_scalar(std::istream& input) {
    T value{};
    input.read(
        reinterpret_cast<char*>(&value),
        static_cast<std::streamsize>(sizeof(T)));
    if (!input) {
        throw std::runtime_error("Truncated physical IC file.");
    }
    return value;
}

// Read the versioned base particle record in its written order. Validate
// physical ranges, exact end of file and the stored fingerprint before use.
std::vector<ev::WeightedPICMarker> read_physical_ic(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "Could not open physical IC file: " + path.string());
    }
    char magic[16]{};
    input.read(magic, sizeof(magic));
    const char expected[16] = {
        'V','E','N','U','S','E','V','P','H','Y','S','I','C','1',0,0,
    };
    if (!input || std::memcmp(magic, expected, sizeof(magic)) != 0) {
        throw std::runtime_error(
            "Physical IC file has the wrong format: " + path.string());
    }
    const std::uint32_t version =
        read_physical_ic_scalar<std::uint32_t>(input);
    const std::uint64_t count =
        read_physical_ic_scalar<std::uint64_t>(input);
    const std::uint64_t expected_fingerprint =
        read_physical_ic_scalar<std::uint64_t>(input);
    if (version != 1
        || count > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            "Physical IC file has an unsupported header: " + path.string());
    }
    std::vector<ev::WeightedPICMarker> markers;
    markers.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t q = 0; q < count; ++q) {
        ev::WeightedPICMarker marker;
        marker.mass = read_physical_ic_scalar<double>(input);
        marker.r = read_physical_ic_scalar<double>(input);
        marker.w = read_physical_ic_scalar<double>(input);
        marker.eta = read_physical_ic_scalar<double>(input);
        marker.id = read_physical_ic_scalar<std::uint64_t>(input);
        marker.source_cell = read_physical_ic_scalar<std::uint64_t>(input);
        if (!(marker.mass > 0.0) || !(marker.r > 0.0)
            || !(marker.eta >= 0.0) || !std::isfinite(marker.mass)
            || !std::isfinite(marker.r) || !std::isfinite(marker.w)
            || !std::isfinite(marker.eta)) {
            throw std::runtime_error(
                "Physical IC file contains an invalid marker: "
                + path.string());
        }
        markers.push_back(marker);
    }
    char trailing = 0;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error(
            "Physical IC file contains trailing data: " + path.string());
    }
    const std::uint64_t fingerprint = physical_ic_fingerprint(markers);
    if (fingerprint != expected_fingerprint) {
        throw std::runtime_error(
            "Physical IC fingerprint mismatch: " + path.string());
    }
    return markers;
}

// Keep an existing identical initial realization. For a new file, finish
// writing all particles to a temporary name before publishing the final name.
void write_physical_ic(
    const std::filesystem::path& path,
    const std::vector<ev::WeightedPICMarker>& markers) {
    if (path.empty()) {
        return;
    }
    const std::uint64_t fingerprint = physical_ic_fingerprint(markers);
    if (std::filesystem::exists(path)) {
        const auto existing = read_physical_ic(path);
        if (physical_ic_fingerprint(existing) != fingerprint) {
            throw std::runtime_error(
                "Refusing to replace a different physical IC file: "
                + path.string());
        }
        return;
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    const auto nonce = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const std::filesystem::path temporary =
        path.string() + ".tmp." + std::to_string(nonce);
    try {
        std::ofstream output(
            temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "Could not create physical IC file: " + temporary.string());
        }
        const char magic[16] = {
            'V','E','N','U','S','E','V','P','H','Y','S','I','C','1',0,0,
        };
        output.write(magic, sizeof(magic));
        write_physical_ic_scalar<std::uint32_t>(output, 1);
        write_physical_ic_scalar<std::uint64_t>(
            output, static_cast<std::uint64_t>(markers.size()));
        write_physical_ic_scalar(output, fingerprint);
        for (const ev::WeightedPICMarker& marker : markers) {
            write_physical_ic_scalar(output, marker.mass);
            write_physical_ic_scalar(output, marker.r);
            write_physical_ic_scalar(output, marker.w);
            write_physical_ic_scalar(output, marker.eta);
            write_physical_ic_scalar(output, marker.id);
            write_physical_ic_scalar(output, marker.source_cell);
        }
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "Could not flush physical IC file: " + temporary.string());
        }
        output.close();
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

// A physical-IC file is an immutable, coordinate-neutral base realization:
// invariant rest weights, areal radii, Schwarzschild orthonormal radial
// momenta, eta, and stable marker IDs.  Angular-momentum controls are applied
// only after this base state is loaded.  Consequently an s_L scan keeps every
// marker weight, radius, radial momentum, and ID fixed while eta -> s_L^2 eta.
std::vector<ev::WeightedPICMarker> load_or_make_high_base_markers(
    const RunOptions& options,
    const ev::Params& parameters,
    const ev::WeightedPICFieldGrid& grid,
    HighSummary* summary = nullptr) {
    std::vector<ev::WeightedPICMarker> markers;
    if (options.high_initial_condition == "energy_cutoff_shell") {
        namespace static_ev = ev::static_einstein_vlasov;
        static_ev::Parameters continuum_parameters;
        continuum_parameters.amplitude =
            options.high_energy_cutoff_amplitude;
        continuum_parameters.angular_momentum =
            options.high_energy_cutoff_ell0;
        continuum_parameters.central_y =
            options.high_energy_cutoff_central_y;
        continuum_parameters.cutoff_power =
            options.high_energy_cutoff_power;
        // Keep the continuum ODE domain independent of the evolution box.
        // The completed shell is checked against the actual WPIC domain
        // below; silently clipping a continuum solution is forbidden.
        continuum_parameters.maximum_radius = std::max(
            continuum_parameters.maximum_radius,
            64.0 * continuum_parameters.angular_momentum);
        // Solve the static Einstein--Vlasov continuum before drawing any
        // particles.  Its energy cutoff fixes compact support and its
        // stress moments determine the self-consistent radial metric.
        static_ev::Solution continuum =
            static_ev::solve(continuum_parameters);
        const double unscaled_adm_mass = continuum.adm_mass;
        if (options.high_target_initial_adm_mass > 0.0) {
            continuum = static_ev::rescale_to_adm_mass(
                continuum, options.high_target_initial_adm_mass);
        }
        if (!(continuum.cutoff_energy > 0.0)
            || !(continuum.cutoff_energy < 1.0)
            || !(continuum.binding_energy > 0.0)
            || !(continuum.binding_fraction > 0.0)
            || !std::isfinite(continuum.binding_fraction)) {
            throw std::runtime_error(
                "energy_cutoff_shell requires a finite bound continuum "
                "branch with E0<1 and M0-M_ADM>0.");
        }
        constexpr std::size_t vacuum_buffer_cells = 4;
        if (grid.mesh.edge.size() <= 2 * vacuum_buffer_cells
            || !(continuum.inner_matter_radius
                > grid.mesh.edge[vacuum_buffer_cells])
            || !(continuum.outer_matter_radius
                < grid.mesh.edge[
                    grid.mesh.edge.size() - 1 - vacuum_buffer_cells])) {
            throw std::runtime_error(
                "The complete energy_cutoff_shell matter support must lie "
                "inside the configured WPIC radial domain with at least four "
                "vacuum cells at each boundary for the TSC stencil.");
        }

        if (options.high_markers % 2 != 0) {
            throw std::runtime_error(
                "energy_cutoff_shell marker count is incompatible with "
                "adjacent shifted-stratified momentum pairs.");
        }
        static_ev::MarkerSamplingOptions sampling;
        sampling.rule = static_ev::MarkerSamplingRule::shifted_stratified;
        sampling.shifted_pair_count = options.high_markers / 2;
        sampling.realization = options.high_sampling_realization;
        const static_ev::MarkerSet sampled =
            static_ev::sample_markers(continuum, sampling);
        if (sampled.markers.size() != options.high_markers) {
            throw std::runtime_error(
                "energy_cutoff_shell sampler did not emit exactly "
                "--high-markers markers.");
        }
        markers = sampled.markers;
        std::unordered_set<std::uint64_t> source_identity;
        source_identity.reserve(markers.size());
        for (std::size_t q = 0; q < markers.size(); q += 2) {
            const ev::WeightedPICMarker& positive = markers[q];
            const ev::WeightedPICMarker& negative = markers[q + 1];
            if (positive.id != positive.source_cell
                || negative.id != negative.source_cell
                || !source_identity.insert(positive.source_cell).second
                || !source_identity.insert(negative.source_cell).second
                || positive.r != negative.r
                || positive.mass != negative.mass
                || positive.eta != negative.eta
                || positive.w != -negative.w) {
                throw std::runtime_error(
                    "energy_cutoff_shell lost its unique persistent source "
                    "IDs or exact adjacent (+w,-w) invariant-weight pairs.");
            }
        }

        // A zero kick retains the exact adjacent (+w,-w) equilibrium pairs.
        // A positive kick gives each pair the same mean -W_in times the same
        // optional core taper used by the Plummer family.  It is applied
        // after continuum quadrature, so invariant rest weights are unchanged.
        for (ev::WeightedPICMarker& marker : markers) {
            double infall_taper = 1.0;
            if (options.high_infall_core_radius > 0.0) {
                const double radius_squared = marker.r * marker.r;
                const double core_squared =
                    options.high_infall_core_radius
                    * options.high_infall_core_radius;
                infall_taper =
                    radius_squared / (radius_squared + core_squared);
            }
            marker.w -= options.high_radial_infall * infall_taper;
        }
        for (std::size_t q = 0; q < markers.size(); q += 2) {
            const double radius = markers[q].r;
            double infall_taper = 1.0;
            if (options.high_infall_core_radius > 0.0) {
                const double radius_squared = radius * radius;
                const double core_squared =
                    options.high_infall_core_radius
                    * options.high_infall_core_radius;
                infall_taper =
                    radius_squared / (radius_squared + core_squared);
            }
            const double pair_mean =
                0.5 * (markers[q].w + markers[q + 1].w);
            const double expected_pair_mean =
                -options.high_radial_infall * infall_taper;
            if (std::abs(pair_mean - expected_pair_mean)
                > 8.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, std::abs(expected_pair_mean))) {
                throw std::runtime_error(
                    "energy_cutoff_shell inward kick did not preserve the "
                    "intended adjacent-pair mean radial momentum.");
            }
        }

        if (summary != nullptr) {
            summary->energy_cutoff_continuum_available = true;
            summary->energy_cutoff_configured_amplitude =
                options.high_energy_cutoff_amplitude;
            summary->energy_cutoff_configured_ell0 =
                options.high_energy_cutoff_ell0;
            summary->energy_cutoff_configured_central_y =
                options.high_energy_cutoff_central_y;
            summary->energy_cutoff_configured_power =
                options.high_energy_cutoff_power;
            summary->energy_cutoff_unscaled_adm_mass = unscaled_adm_mass;
            summary->energy_cutoff_realized_amplitude =
                continuum.parameters.amplitude;
            summary->energy_cutoff_realized_ell0 =
                continuum.parameters.angular_momentum;
            summary->energy_cutoff_realized_power =
                continuum.parameters.cutoff_power;
            summary->energy_cutoff_cutoff_energy = continuum.cutoff_energy;
            summary->energy_cutoff_inner_matter_radius =
                continuum.inner_matter_radius;
            summary->energy_cutoff_outer_matter_radius =
                continuum.outer_matter_radius;
            summary->energy_cutoff_continuum_adm_mass = continuum.adm_mass;
            summary->energy_cutoff_continuum_rest_weight =
                continuum.rest_weight;
            summary->energy_cutoff_continuum_binding_energy =
                continuum.binding_energy;
            summary->energy_cutoff_continuum_binding_fraction =
                continuum.binding_fraction;
            summary->energy_cutoff_central_lapse = continuum.central_lapse;
            summary->energy_cutoff_maximum_compactness =
                continuum.diagnostics.maximum_compactness;
            summary->energy_cutoff_minimum_compactness_gap =
                continuum.diagnostics.minimum_compactness_gap;
            summary->energy_cutoff_constraint_residual_linf =
                continuum.diagnostics.constraint_residual_linf;
            summary->energy_cutoff_jeans_residual_relative_l2 =
                continuum.diagnostics.jeans_residual_relative_l2;
            summary->energy_cutoff_sampled_rest_weight =
                sampled.represented_rest_weight;
            summary->energy_cutoff_sampled_rest_weight_relative_error =
                sampled.rest_weight_relative_error;
            summary->energy_cutoff_sampled_maximum_energy_over_cutoff =
                sampled.maximum_energy_over_cutoff;
            summary->energy_cutoff_sampling_pair_count =
                sampling.shifted_pair_count;
        }

        // Reconstructing the expected deterministic realization even when a
        // file exists authenticates every continuum and sampling option.  It
        // also prevents a stale physical IC from masquerading as a resume or
        // null-map member of a different static family.
        if (!options.physical_ic_path.empty()
            && std::filesystem::exists(options.physical_ic_path)) {
            const auto stored = read_physical_ic(options.physical_ic_path);
            if (physical_ic_fingerprint(stored)
                != physical_ic_fingerprint(markers)) {
                throw std::runtime_error(
                    "Stored energy_cutoff_shell physical IC does not match "
                    "the configured continuum, kick, sampling orders, or "
                    "realization.");
            }
            markers = stored;
        } else {
            write_physical_ic(options.physical_ic_path, markers);
        }
    } else if (!options.physical_ic_path.empty()
               && std::filesystem::exists(options.physical_ic_path)) {
        markers = read_physical_ic(options.physical_ic_path);
    } else {
        if (options.high_initial_condition == "plummer") {
            markers = make_cold_plummer_markers(
                options.high_markers,
                options.high_total_mass,
                options.high_scale_radius,
                grid.mesh.edge.front(),
                options.high_r_max,
                options.high_velocity_factor,
                options.high_radial_importance_exponent,
                options.high_sampling_realization);
            for (auto& marker : markers) {
                double infall_taper = 1.0;
                if (options.high_infall_core_radius > 0.0) {
                    const double radius_squared = marker.r * marker.r;
                    const double core_squared =
                        options.high_infall_core_radius
                        * options.high_infall_core_radius;
                    infall_taper =
                        radius_squared / (radius_squared + core_squared);
                }
                marker.w -= options.high_radial_infall * infall_taper;
            }
        } else {
            markers = make_infalling_shell_markers(
                options.high_markers,
                options.high_total_mass,
                grid.mesh.edge.front(),
                options.high_r_max,
                options.high_shell_center,
                options.high_shell_width,
                options.high_shell_radial_importance_factor,
                options.high_shell_radial_momentum,
                options.high_shell_radial_momentum_dispersion,
                options.high_shell_tangential_velocity_dispersion,
                options.high_sampling_realization);
        }
        // The Plummer/Gaussian samplers target a coordinate-volume
        // density.  Canonicalization chooses invariant rest weights once for
        // those base realizations.  The energy-cutoff branch above bypasses
        // this call because its continuum quadrature already returned the
        // exact invariant Schwarzschild WPIC measure.
        ev::canonicalize_weighted_einstein_vlasov_pic_initial_weights(
            markers, parameters);
        write_physical_ic(options.physical_ic_path, markers);
    }
    if (markers.size() != options.high_markers) {
        throw std::runtime_error(
            "Physical/base IC marker count does not match --high-markers.");
    }
    return markers;
}

// Sort pointers by CURRENT radius, then integrate invariant rest weight to
// obtain enclosed-fraction radii. Particle identities and records stay in
// place. K and W below are Newtonian quadratic-momentum and shell-energy
// proxies; the relativistic equilibrium test separately uses Jeans balance.
RadialDiagnostics radial_diagnostics(
    const std::vector<ev::WeightedPICMarker>& markers,
    double gravity = 1.0) {
    std::vector<const ev::WeightedPICMarker*> active;
    active.reserve(markers.size());
    for (const auto& value : markers) {
        if (value.state == ev::WeightedPICMarkerState::active) {
            active.push_back(&value);
        }
    }
    if (active.empty()) {
        throw std::runtime_error("No active markers in radial diagnostics.");
    }
    std::sort(
        active.begin(),
        active.end(),
        [](const auto* first, const auto* second) {
            if (first->r != second->r) {
                return first->r < second->r;
            }
            return first->id < second->id;
        });
    long double total_mass = 0.0L;
    for (const auto* value : active) {
        total_mass += value->mass;
    }
    auto quantile = [&](double fraction) {
        const long double target = fraction * total_mass;
        long double enclosed = 0.0L;
        for (const auto* value : active) {
            enclosed += value->mass;
            if (enclosed >= target) {
                return value->r;
            }
        }
        return active.back()->r;
    };

    RadialDiagnostics result;
    result.r10 = quantile(0.1);
    result.r50 = quantile(0.5);
    result.r90 = quantile(0.9);
    long double enclosed = 0.0L;
    long double kinetic = 0.0L;
    long double potential = 0.0L;
    for (const auto* value : active) {
        const long double vt2 =
            2.0L * value->eta / (value->r * value->r);
        kinetic += 0.5L * value->mass
            * (value->w * value->w + vt2);
        // Enclosed particles supply the inner-shell interaction. Half this
        // shell's own mass gives its spherical self-energy contribution.
        potential -= gravity * value->mass
            * (enclosed + 0.5L * value->mass) / value->r;
        enclosed += value->mass;
    }
    result.kinetic = static_cast<double>(kinetic);
    result.potential = static_cast<double>(potential);
    result.virial_ratio = result.potential < 0.0
        ? 2.0 * result.kinetic / std::abs(result.potential)
        : std::numeric_limits<double>::infinity();
    return result;
}

double shell_mass_l1_difference(
    const std::vector<ev::WeightedPICMarker>& first,
    const std::vector<ev::WeightedPICMarker>& second,
    const ev::WeightedPICRadialMesh& mesh,
    ev::WeightedPICParticleShape shape) {
    const auto first_deposit =
        ev::deposit_weighted_vlasov_pic(first, mesh, shape);
    const auto second_deposit =
        ev::deposit_weighted_vlasov_pic(second, mesh, shape);
    double difference = 0.0;
    double normalization = 0.0;
    for (std::size_t i = 0; i < first_deposit.shell_mass.size(); ++i) {
        difference += std::abs(
            first_deposit.shell_mass[i]
            - second_deposit.shell_mass[i]);
        normalization += 0.5 * (
            first_deposit.shell_mass[i]
            + second_deposit.shell_mass[i]);
    }
    return difference / normalization;
}

std::pair<double, double> particle_rms_difference(
    const std::vector<ev::WeightedPICMarker>& first,
    const std::vector<ev::WeightedPICMarker>& second) {
    if (first.size() != second.size()) {
        throw std::runtime_error("Particle comparison size mismatch.");
    }
    long double radial = 0.0L;
    long double momentum = 0.0L;
    for (std::size_t q = 0; q < first.size(); ++q) {
        radial += std::pow(first[q].r - second[q].r, 2);
        momentum += std::pow(first[q].w - second[q].w, 2);
    }
    return {
        std::sqrt(static_cast<double>(radial / first.size())),
        std::sqrt(static_cast<double>(momentum / first.size())),
    };
}

// ---- CSV profile/history schemas -----------------------------------------
void write_low_row(
    std::ofstream& output,
    int step,
    double time,
    double dt,
    const ev::WeightedEinsteinVlasovPICField& ev_field,
    const ev::VlasovPoissonField& vp_field,
    const RadialDiagnostics& ev_radial,
    const RadialDiagnostics& vp_radial,
    double density_l1,
    double rms_r,
    double rms_w) {
    output << step << ',' << time << ',' << dt << ','
           << ev_field.rest_mass << ',' << ev_field.adm_mass << ','
           << ev_field.maximum_compactness << ','
           << ev_field.metric.min_gap << ','
           << ev_radial.r10 << ',' << ev_radial.r50 << ','
           << ev_radial.r90 << ',' << ev_radial.virial_ratio << ','
           << vp_field.matter_mass << ','
           << vp_radial.r10 << ',' << vp_radial.r50 << ','
           << vp_radial.r90 << ',' << vp_radial.virial_ratio << ','
           << density_l1 << ',' << rms_r << ',' << rms_w << '\n';
}

void write_low_profile(
    std::ofstream& output,
    int step,
    double time,
    const ev::WeightedPICFieldGrid& grid,
    const ev::WeightedEinsteinVlasovPICField& ev_field,
    const ev::WeightedPICDepositedMoments& vp_deposit,
    const ev::VlasovPoissonField& vp_field) {
    const std::size_t cells = grid.radius.size();
    if (ev_field.moments.rho.size() != cells
        || ev_field.metric.m.size() != cells
        || vp_deposit.density.size() != cells
        || vp_field.source_mass.size() != cells) {
        throw std::runtime_error(
            "Low-compactness profile output received inconsistent fields.");
    }
    for (std::size_t i = 0; i < cells; ++i) {
        const double radius = grid.radius[i];
        output << step << ',' << time << ',' << radius << ','
               << ev_field.moments.rho[i] << ','
               << 2.0 * ev_field.metric.m[i] / radius << ','
               << vp_deposit.density[i] << ','
               << 2.0 * vp_field.source_mass[i] / radius << '\n';
    }
}

void write_high_profile(
    std::ofstream& output,
    int step,
    double time,
    const ev::WeightedPICFieldGrid& grid,
    const ev::WeightedEinsteinVlasovPICField& field) {
    const std::size_t cells = grid.radius.size();
    if (field.moments.rho.size() != cells
        || field.moments.p.size() != cells
        || field.moments.pt.size() != cells
        || field.moments.j.size() != cells
        || field.metric.m.size() != cells
        || field.metric.lambda.size() != cells
        || field.metric.mu.size() != cells
        || field.metric.mu_r.size() != cells
        || field.metric.lambda_t.size() != cells
        || field.metric.face_radius.size() != cells + 1
        || field.metric.face_lambda.size() != cells + 1
        || field.metric.face_mu.size() != cells + 1
        || field.metric.face_mu_r.size() != cells + 1) {
        throw std::runtime_error(
            "High-compactness profile output received inconsistent fields.");
    }
    for (std::size_t i = 0; i < cells; ++i) {
        const double radius = grid.radius[i];
        output << step << ',' << time << ',' << radius << ','
               << field.moments.rho[i] << ','
               << 2.0 * field.metric.m[i] / radius << ','
               << field.metric.m[i] << ','
               << field.metric.gap[i] << ','
               << field.metric.lambda[i] << ','
               << field.metric.mu[i] << ','
               << field.metric.lambda_t[i] << ','
               << field.moments.p[i] << ','
               << field.moments.pt[i] << ','
               << field.moments.j[i]
               << ",cell_center," << i << ','
               << field.metric.face_radius[i] << ','
               << field.metric.face_radius[i + 1] << ','
               << field.metric.face_lambda[i] << ','
               << field.metric.face_lambda[i + 1] << ','
               << field.metric.mu_r[i] << ','
               << field.metric.face_mu[i] << ','
               << field.metric.face_mu[i + 1] << ','
               << field.metric.face_mu_r[i] << ','
               << field.metric.face_mu_r[i + 1] << '\n';
    }
    // The dynamic eikonal is normalized on the authenticated outer vacuum
    // worldtube, not at the last cell center.  Save that exact face so a
    // parity tool never has to extrapolate the coordinate transformation.
    if (field.metric.face_radius.size() == cells + 1
        && field.metric.face_m.size() == cells + 1
        && field.metric.face_gap.size() == cells + 1
        && field.metric.face_lambda.size() == cells + 1
        && field.metric.face_mu.size() == cells + 1
        && field.metric.face_lambda_t.size() == cells + 1) {
        const std::size_t face = cells;
        const double radius = field.metric.face_radius[face];
        output << step << ',' << time << ',' << radius << ",0,"
               << 2.0 * field.metric.face_m[face] / radius << ','
               << field.metric.face_m[face] << ','
               << field.metric.face_gap[face] << ','
               << field.metric.face_lambda[face] << ','
               << field.metric.face_mu[face] << ','
               << field.metric.face_lambda_t[face]
               << ",0,0,0,outer_face," << cells << ','
               << radius << ',' << radius << ','
               << field.metric.face_lambda[face] << ','
               << field.metric.face_lambda[face] << ','
               << field.metric.face_mu_r[face] << ','
               << field.metric.face_mu[face] << ','
               << field.metric.face_mu[face] << ','
               << field.metric.face_mu_r[face] << ','
               << field.metric.face_mu_r[face] << '\n';
    }
}

struct EFCovariantStressAudit {
    std::vector<double> t_vv;
    std::vector<double> t_vr;
    std::vector<double> t_theta_theta;
    std::vector<double> rho_infall;
};

// Reconstruct additional EF stress components from the same active markers
// and metric as the field solve.  Here P=marker.p_r>0 is covariant radial
// momentum, chi=1+2*eta/r^2 and p_v=-b*(a*P+chi/P)/2 follows from the mass
// shell.  The null-slice momentum measure gives the factor 1/P in T_ab.
EFCovariantStressAudit deposit_ef_covariant_stress_audit(
    const std::vector<ev::eddington_finkelstein::Marker>& markers,
    const ev::WeightedPICFieldGrid& grid,
    const ev::eddington_finkelstein::Metric& metric) {
    const std::size_t cells = grid.radius.size();
    std::vector<long double> t_vv(cells, 0.0L);
    std::vector<long double> t_vr(cells, 0.0L);
    std::vector<long double> t_theta_theta(cells, 0.0L);
    std::vector<long double> rho_infall(cells, 0.0L);
    const ev::eddington_finkelstein::MetricSampler metric_sampler(grid, metric);
    for (const auto& marker : markers) {
        if (marker.state
            != ev::eddington_finkelstein::MarkerState::active) {
            continue;
        }
        const auto local = metric_sampler(marker.r);
        const long double radius = std::max(
            static_cast<long double>(marker.r), 1.0e-150L);
        const long double p_r = marker.p_r;
        const long double chi = 1.0L
            + 2.0L * marker.eta / (radius * radius);
        const long double p_v = -0.5L * local.b
            * (static_cast<long double>(local.a) * p_r + chi / p_r);
        const long double weight = marker.rest_weight;
        const long double marker_t_vv = weight * p_v * p_v / p_r;
        const long double marker_t_vr = weight * p_v;
        const long double marker_t_theta_theta =
            weight * marker.eta / p_r;
        const long double energy = ev::eddington_finkelstein::infall_observer_energy(
            marker.r, marker.p_r, marker.eta);
        // Contract the stress with the prescribed inward observer twice.
        // Its measured energy replaces the coordinate momentum component
        // while the same 1/P slice measure remains.
        const long double marker_rho_infall = weight * energy * energy / p_r;
        // Spread each contribution with the same TSC shape as the source
        // deposition.  Dividing by shell volume below converts its weighted
        // sum into a cell-averaged stress component.
        const ev::WeightedPICTSCStencil stencil =
            ev::weighted_pic_tsc_stencil(marker.r, grid.mesh);
        for (int n = 0; n < stencil.count; ++n) {
            const std::size_t slot = static_cast<std::size_t>(n);
            const std::size_t cell = stencil.cell[slot];
            const long double shape_weight = stencil.weight[slot];
            t_vv[cell] += shape_weight * marker_t_vv;
            t_vr[cell] += shape_weight * marker_t_vr;
            t_theta_theta[cell] +=
                shape_weight * marker_t_theta_theta;
            rho_infall[cell] += shape_weight * marker_rho_infall;
        }
    }
    EFCovariantStressAudit result;
    result.t_vv.resize(cells);
    result.t_vr.resize(cells);
    result.t_theta_theta.resize(cells);
    result.rho_infall.resize(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        const long double inverse_volume =
            1.0L / grid.mesh.shell_volume[i];
        result.t_vv[i] = static_cast<double>(t_vv[i] * inverse_volume);
        result.t_vr[i] = static_cast<double>(t_vr[i] * inverse_volume);
        result.t_theta_theta[i] = static_cast<double>(
            t_theta_theta[i] * inverse_volume);
        result.rho_infall[i] = static_cast<double>(rho_infall[i] * inverse_volume);
        if (!std::isfinite(result.rho_infall[i]) || result.rho_infall[i] < 0.0) {
            throw std::runtime_error("EF infall density is nonfinite or negative.");
        }
    }
    return result;
}

// This reports the last accepted slice without changing its field or markers.
// A regular origin has m(0)=0.  An inferred central mass residual
// comparable to the entire ADM mass establishes lost spatial resolution.
// Smaller residuals still require the science case's convergence comparison.
void record_ef_spatial_resolution(
    HighSummary& summary,
    const std::vector<ev::eddington_finkelstein::Marker>& markers,
    const ev::WeightedPICFieldGrid& grid,
    const ev::eddington_finkelstein::Field& field,
    bool enable_excision) {
    const double first_width = grid.mesh.edge[1] - grid.mesh.edge[0];
    if (summary.ef_final_active_min_radius >= 0.0 && first_width > 0.0) {
        summary.ef_minimum_active_radius_over_first_cell_width =
            summary.ef_final_active_min_radius / first_width;
    }
    if (enable_excision) {
        summary.ef_spatial_resolution_assessment = "excision_enabled";
    } else if (grid.mesh.edge.front() != 0.0) {
        summary.ef_spatial_resolution_assessment = "finite_inner_boundary";
    } else if (field.metric.adm_mass > 0.0) {
        summary.ef_origin_mass_residual_available = true;
        const long double fraction = std::abs(
            static_cast<long double>(field.metric.face_m.front()))
            / field.metric.adm_mass;
        summary.ef_origin_mass_residual_over_adm = static_cast<double>(
            std::min(fraction,
                static_cast<long double>(std::numeric_limits<double>::max())));
        summary.ef_spatial_resolution_assessment = fraction >= 1.0L
            ? "spatial_resolution_lost" : "requires_convergence";
    } else {
        summary.ef_spatial_resolution_assessment = "zero_adm_mass";
    }
    const auto stress = deposit_ef_covariant_stress_audit(
        markers, grid, field.metric);
    const auto retained = [&](std::size_t i) {
        return !(field.metric.excision_radius > 0.0
            && grid.radius[i] <= field.metric.excision_radius);
    };
    std::size_t peak = 0;
    double maximum = 0.0;
    for (std::size_t i = 0; i < grid.radius.size(); ++i) {
        if (retained(i) && stress.rho_infall[i] > maximum) {
            maximum = stress.rho_infall[i];
            peak = i;
        }
    }
    // Count neighboring resolved cells above half the peak infalling-frame
    // density.  This records how many cells represent the densest structure.
    if (maximum > 0.0) {
        std::size_t left = peak, right = peak;
        while (left > 0 && retained(left - 1)
            && stress.rho_infall[left - 1] >= 0.5 * maximum) --left;
        while (right + 1 < grid.radius.size() && retained(right + 1)
            && stress.rho_infall[right + 1] >= 0.5 * maximum) ++right;
        summary.ef_density_peak_radius = grid.radius[peak];
        summary.ef_density_peak_half_maximum_cells = right - left + 1;
    }
}

struct AnisotropicJeansDiagnostics {
    double relative_l1 = -1.0;
    int supported_cells = 0;
};

// Static spherical stress balance in polar-areal variables:
//   p_r' + (rho+p_r) mu_r - (p_t_sum-2p_r)/r = 0.
// The normalized L1 residual is meaningful as a late-time stationarity test,
// not as an evolution equation.  We exclude the two derivative boundary
// cells and density-floor cells, where a PIC pressure derivative is
// unresolved.  Vacuum is excluded by a relative rho threshold.
AnisotropicJeansDiagnostics anisotropic_jeans_relative_l1(
    const std::vector<double>& radius,
    const std::vector<double>& shell_volume,
    const std::vector<double>& rho,
    const std::vector<double>& p_radial,
    const std::vector<double>& p_tangential_sum,
    const std::vector<double>& mu_r,
    const std::vector<bool>& valid) {
    const std::size_t cells = radius.size();
    if (cells < 3 || shell_volume.size() != cells || rho.size() != cells
        || p_radial.size() != cells || p_tangential_sum.size() != cells
        || mu_r.size() != cells || valid.size() != cells) {
        throw std::runtime_error(
            "Anisotropic Jeans diagnostic received inconsistent arrays.");
    }
    double maximum_rho = 0.0;
    for (std::size_t i = 0; i < cells; ++i) {
        if (valid[i] && std::isfinite(rho[i])) {
            maximum_rho = std::max(maximum_rho, rho[i]);
        }
    }
    if (!(maximum_rho > 0.0)) {
        return {};
    }
    const double density_floor = 1.0e-6 * maximum_rho;
    long double numerator = 0.0L;
    long double denominator = 0.0L;
    int supported_cells = 0;
    for (std::size_t i = 1; i + 1 < cells; ++i) {
        if (!valid[i - 1] || !valid[i] || !valid[i + 1]
            || !(rho[i - 1] > density_floor)
            || !(rho[i] > density_floor)
            || !(rho[i + 1] > density_floor)
            || !(radius[i] > 0.0)) {
            continue;
        }
        const double width = radius[i + 1] - radius[i - 1];
        if (!(width > 0.0)) {
            continue;
        }
        const double pressure_gradient =
            (p_radial[i + 1] - p_radial[i - 1]) / width;
        const double gravity = (rho[i] + p_radial[i]) * mu_r[i];
        const double anisotropy =
            (p_tangential_sum[i] - 2.0 * p_radial[i]) / radius[i];
        const double residual = pressure_gradient + gravity - anisotropy;
        const double magnitude = std::abs(pressure_gradient)
            + std::abs(gravity) + std::abs(anisotropy);
        if (!std::isfinite(residual) || !std::isfinite(magnitude)) {
            continue;
        }
        const long double weight = shell_volume[i];
        numerator += weight * std::abs(residual);
        denominator += weight * magnitude;
        ++supported_cells;
    }
    if (supported_cells < 3 || !(denominator > 0.0L)) {
        return {};
    }
    return {
        static_cast<double>(numerator / denominator),
        supported_cells,
    };
}

AnisotropicJeansDiagnostics schwarzschild_jeans_diagnostics(
    const ev::WeightedPICFieldGrid& grid,
    const ev::WeightedEinsteinVlasovPICField& field) {
    const std::size_t cells = grid.radius.size();
    std::vector<bool> valid(cells, true);
    for (std::size_t i = 0; i < cells; ++i) {
        valid[i] = field.metric.gap[i] > 0.0
            && std::isfinite(field.moments.rho[i])
            && std::isfinite(field.moments.p[i])
            && std::isfinite(field.moments.pt[i])
            && std::isfinite(field.metric.mu_r[i]);
    }
    return anisotropic_jeans_relative_l1(
        grid.radius,
        grid.mesh.shell_volume,
        field.moments.rho,
        field.moments.p,
        field.moments.pt,
        field.metric.mu_r,
        valid);
}

// In the exterior a>0, transform the EF stress at each event to the local
// static frame.  That supplies rho, radial pressure and total tangential
// pressure for the same spherical Jeans balance evaluated in SC variables.
// The static frame is available only in the exterior, as checked below.
AnisotropicJeansDiagnostics ef_jeans_diagnostics(
    const std::vector<ev::eddington_finkelstein::Marker>& markers,
    const ev::WeightedPICFieldGrid& grid,
    const ev::eddington_finkelstein::Field& field) {
    const std::size_t cells = grid.radius.size();
    const EFCovariantStressAudit stress =
        deposit_ef_covariant_stress_audit(markers, grid, field.metric);
    std::vector<double> rho(cells, 0.0);
    std::vector<double> p_radial(cells, 0.0);
    std::vector<double> p_tangential_sum(cells, 0.0);
    std::vector<double> mu_r(cells, 0.0);
    std::vector<bool> valid(cells, false);
    for (std::size_t i = 0; i < cells; ++i) {
        const double radius = grid.radius[i];
        const double a = field.metric.a[i];
        const double b = field.metric.b[i];
        if (!(radius > 0.0) || !(a > 0.0) || !(b > 0.0)) {
            continue;
        }
        // Exact same-event polar moment pullback in the exterior:
        // rho=T_vv/(a b^2), j=-T_vr/b-rho, p_r=rho-S.
        rho[i] = stress.t_vv[i] / (a * b * b);
        p_radial[i] = rho[i] - field.sources.s[i];
        p_tangential_sum[i] =
            2.0 * stress.t_theta_theta[i] / (radius * radius);
        mu_r[i] = (
            field.metric.m[i]
            + 4.0 * ev::PI * radius * radius * radius * p_radial[i]
        ) / (radius * radius * a);
        valid[i] = std::isfinite(rho[i])
            && std::isfinite(p_radial[i])
            && std::isfinite(p_tangential_sum[i])
            && std::isfinite(mu_r[i]);
    }
    return anisotropic_jeans_relative_l1(
        grid.radius,
        grid.mesh.shell_volume,
        rho,
        p_radial,
        p_tangential_sum,
        mu_r,
        valid);
}

void write_ef_high_profile(
    std::ofstream& output,
    int step,
    double advanced_time,
    const std::vector<ev::eddington_finkelstein::Marker>& markers,
    const ev::WeightedPICFieldGrid& grid,
    const ev::eddington_finkelstein::Field& field) {
    const std::size_t cells = grid.radius.size();
    if (field.metric.a.size() != cells
        || field.metric.b.size() != cells
        || field.metric.m.size() != cells
        || field.metric.face_radius.size() != cells + 1
        || field.metric.face_a.size() != cells + 1
        || field.metric.face_b.size() != cells + 1
        || field.sources.t_rr.size() != cells
        || field.sources.s.size() != cells
        || field.sources.shell_rest_weight.size() != cells) {
        throw std::runtime_error(
            "EF high-compactness profile received inconsistent fields.");
    }
    const EFCovariantStressAudit stress =
        deposit_ef_covariant_stress_audit(markers, grid, field.metric);
    for (std::size_t i = 0; i < cells; ++i) {
        output << step << ',' << advanced_time << ',' << grid.radius[i]
               << ',' << field.metric.a[i]
               << ',' << field.metric.b[i]
               << ',' << field.metric.m[i]
               << ',' << field.metric.compactness[i]
               << ',' << field.sources.t_rr[i]
               << ',' << field.sources.s[i]
               << ',' << stress.t_vv[i]
               << ',' << stress.t_vr[i]
               << ',' << stress.t_theta_theta[i]
               << ',' << field.sources.shell_rest_weight[i]
                    / grid.mesh.shell_volume[i]
               << ",cell_center," << i << ','
               << field.metric.face_radius[i] << ','
               << field.metric.face_radius[i + 1] << ','
               << field.metric.face_a[i] << ','
               << field.metric.face_a[i + 1] << ','
               << field.metric.face_b[i] << ','
               << field.metric.face_b[i + 1] << ','
               << stress.rho_infall[i] << ','
               << (field.metric.excision_radius > 0.0
                   && grid.radius[i] <= field.metric.excision_radius ? 1 : 0) << '\n';
    }
    if (field.metric.face_radius.size() == cells + 1
        && field.metric.face_a.size() == cells + 1
        && field.metric.face_b.size() == cells + 1
        && field.metric.face_m.size() == cells + 1
        && field.metric.face_compactness.size() == cells + 1) {
        const std::size_t face = cells;
        output << step << ',' << advanced_time << ','
               << field.metric.face_radius[face] << ','
               << field.metric.face_a[face] << ','
               << field.metric.face_b[face] << ','
               << field.metric.face_m[face] << ','
               << field.metric.face_compactness[face]
               << ",0,0,0,0,0,0,outer_face," << cells << ','
               << field.metric.face_radius[face] << ','
               << field.metric.face_radius[face] << ','
               << field.metric.face_a[face] << ','
               << field.metric.face_a[face] << ','
               << field.metric.face_b[face] << ','
               << field.metric.face_b[face] << ",0,0\n";
    }
}

// Sort pointers to active markers by their current radii, then accumulate
// invariant rest weights to locate mass quantiles.  The K/W sum below is a
// Newtonian diagnostic in the exterior static frame.  Its availability is
// recorded separately from the fully relativistic stress diagnostics.
RadialDiagnostics ef_radial_diagnostics(
    const std::vector<ev::eddington_finkelstein::Marker>& markers,
    const ev::WeightedPICFieldGrid& grid,
    const ev::eddington_finkelstein::Metric& metric,
    double gravity = 1.0) {
    using EFMarker = ev::eddington_finkelstein::Marker;
    std::vector<const EFMarker*> active;
    active.reserve(markers.size());
    for (const EFMarker& marker : markers) {
        if (marker.state
            == ev::eddington_finkelstein::MarkerState::active) {
            active.push_back(&marker);
        }
    }
    RadialDiagnostics result;
    if (active.empty()) {
        result.virial_ratio = -1.0;
        return result;
    }
    std::sort(
        active.begin(),
        active.end(),
        [](const EFMarker* first, const EFMarker* second) {
            if (first->r != second->r) {
                return first->r < second->r;
            }
            return first->id < second->id;
        });
    long double total_rest_weight = 0.0L;
    for (const EFMarker* marker : active) {
        total_rest_weight += marker->rest_weight;
    }
    auto quantile = [&](double fraction) {
        const long double target = fraction * total_rest_weight;
        long double enclosed = 0.0L;
        for (const EFMarker* marker : active) {
            enclosed += marker->rest_weight;
            if (enclosed >= target) {
                return marker->r;
            }
        }
        return active.back()->r;
    };
    result.r10 = quantile(0.1);
    result.r50 = quantile(0.5);
    result.r90 = quantile(0.9);

    long double enclosed = 0.0L;
    long double kinetic = 0.0L;
    long double potential = 0.0L;
    bool exterior_orthonormal_frame_available = true;
    const ev::eddington_finkelstein::MetricSampler metric_sampler(grid, metric);
    for (const EFMarker* marker : active) {
        const auto sampled = metric_sampler(marker->r);
        if (!(sampled.a > 0.0) || !(marker->r > 0.0)) {
            exterior_orthonormal_frame_available = false;
        } else {
            const double w =
                ev::eddington_finkelstein::
                    schwarzschild_w_from_canonical_p_r(
                        marker->p_r,
                        marker->eta,
                        marker->r,
                        sampled.a);
            const long double vt2 =
                2.0L * marker->eta / (marker->r * marker->r);
            kinetic += 0.5L * marker->rest_weight
                * (w * w + vt2);
        }
        potential -= gravity * marker->rest_weight
            * (enclosed + 0.5L * marker->rest_weight) / marker->r;
        enclosed += marker->rest_weight;
    }
    result.kinetic = exterior_orthonormal_frame_available
        ? static_cast<double>(kinetic) : -1.0;
    result.potential = static_cast<double>(potential);
    result.virial_ratio = exterior_orthonormal_frame_available
            && result.potential < 0.0
        ? 2.0 * result.kinetic / std::abs(result.potential)
        : -1.0;
    return result;
}

// ---- Resume-safe CSV and restart handling --------------------------------
std::vector<double> parse_numeric_csv_row(
    const std::string& line,
    std::size_t expected_columns,
    const std::filesystem::path& path) {
    std::vector<double> values;
    values.reserve(expected_columns);
    std::istringstream input(line);
    std::string field;
    while (std::getline(input, field, ',')) {
        try {
            values.push_back(std::stod(field));
        } catch (const std::exception&) {
            throw std::runtime_error(
                "Invalid numeric field in restart history: "
                + path.string());
        }
    }
    if (values.size() != expected_columns) {
        throw std::runtime_error(
            "Unexpected column count in restart history: "
            + path.string());
    }
    return values;
}

std::vector<std::vector<double>> read_numeric_csv_history(
    const std::filesystem::path& path,
    std::size_t expected_columns) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "Could not open restart history: " + path.string());
    }
    std::string line;
    if (!std::getline(input, line) || line.empty()) {
        throw std::runtime_error(
            "Restart history has no header: " + path.string());
    }
    std::vector<std::vector<double>> rows;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            rows.push_back(
                parse_numeric_csv_row(line, expected_columns, path));
        }
    }
    if (rows.empty()) {
        throw std::runtime_error(
            "Restart history has no data rows: " + path.string());
    }
    return rows;
}

std::filesystem::path fixed_family_bound_diagnostics_path(
    const std::filesystem::path& output_dir) {
    return output_dir / "fixed_family_initial_bound_diagnostics.csv";
}

// Check the two binding measures together: rest mass minus ADM mass and
// that difference divided by rest mass.  The frozen-metric energy proxy
// exp(mu)*epsilon also records which initial particles have energy >=1.
// Its values characterize this initial state in the specified metric.
void validate_fixed_family_bound_diagnostics(
    const FixedFamilyBoundDiagnostics& value,
    const char* context) {
    const double scale = std::max({
        1.0, std::abs(value.rest_weight), std::abs(value.adm_mass),
        std::abs(value.binding_energy),
    });
    const double tolerance = 512.0
        * std::numeric_limits<double>::epsilon() * scale;
    if (!(value.rest_weight > 0.0) || !(value.adm_mass > 0.0)
        || !std::isfinite(value.binding_fraction)
        || std::abs(value.binding_energy
                - (value.rest_weight - value.adm_mass)) > tolerance
        || std::abs(value.binding_fraction
                - value.binding_energy / value.rest_weight)
            > tolerance
        || !(value.energy_proxy_minimum > 0.0)
        || !(value.energy_proxy_maximum >= value.energy_proxy_minimum)
        || !(value.energy_proxy_rest_weighted_mean
                >= value.energy_proxy_minimum - tolerance)
        || !(value.energy_proxy_rest_weighted_mean
                <= value.energy_proxy_maximum + tolerance)
        || !(value.energy_proxy_rest_weight_fraction_ge_one >= 0.0)
        || !(value.energy_proxy_rest_weight_fraction_ge_one <= 1.0)
        || !std::isfinite(value.energy_proxy_maximum)
        || !std::isfinite(value.energy_proxy_rest_weighted_mean)) {
        throw std::runtime_error(
            std::string(context)
            + " contains inconsistent fixed-family bound diagnostics.");
    }
}

void write_fixed_family_bound_diagnostics(
    const std::filesystem::path& output_dir,
    const FixedFamilyBoundDiagnostics& value) {
    validate_fixed_family_bound_diagnostics(
        value, "Fixed-family diagnostic write");
    const std::filesystem::path path =
        fixed_family_bound_diagnostics_path(output_dir);
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Could not create fixed-family bound diagnostics: "
            + path.string());
    }
    output << "fixed_family_rest_weight,fixed_family_adm_mass,"
              "fixed_family_binding_energy,fixed_family_binding_fraction,"
              "initial_energy_at_infinity_proxy_minimum,"
              "initial_energy_at_infinity_proxy_maximum,"
              "initial_energy_at_infinity_proxy_rest_weighted_mean,"
              "initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one\n"
           << std::setprecision(17)
           << value.rest_weight << ',' << value.adm_mass << ','
           << value.binding_energy << ',' << value.binding_fraction << ','
           << value.energy_proxy_minimum << ','
           << value.energy_proxy_maximum << ','
           << value.energy_proxy_rest_weighted_mean << ','
           << value.energy_proxy_rest_weight_fraction_ge_one << '\n';
    ev::flush_output_checked(
        output, "fixed-family initial bound diagnostics");
}

FixedFamilyBoundDiagnostics read_fixed_family_bound_diagnostics(
    const std::filesystem::path& output_dir) {
    const std::filesystem::path path =
        fixed_family_bound_diagnostics_path(output_dir);
    const auto rows = read_numeric_csv_history(path, 8);
    if (rows.size() != 1) {
        throw std::runtime_error(
            "Fixed-family bound diagnostic sidecar must have exactly one "
            "data row: " + path.string());
    }
    const auto& row = rows.front();
    FixedFamilyBoundDiagnostics value;
    value.rest_weight = row[0];
    value.adm_mass = row[1];
    value.binding_energy = row[2];
    value.binding_fraction = row[3];
    value.energy_proxy_minimum = row[4];
    value.energy_proxy_maximum = row[5];
    value.energy_proxy_rest_weighted_mean = row[6];
    value.energy_proxy_rest_weight_fraction_ge_one = row[7];
    validate_fixed_family_bound_diagnostics(
        value, "Fixed-family diagnostic resume sidecar");
    return value;
}

int last_csv_step(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "Could not open restart CSV: " + path.string());
    }
    std::string line;
    std::string last_line;
    if (!std::getline(input, line)) {
        throw std::runtime_error(
            "Restart CSV has no header: " + path.string());
    }
    while (std::getline(input, line)) {
        if (!line.empty()) {
            last_line = line;
        }
    }
    if (last_line.empty()) {
        return -1;
    }
    const std::size_t comma = last_line.find(',');
    if (comma == std::string::npos) {
        throw std::runtime_error(
            "Restart CSV has a malformed data row: " + path.string());
    }
    try {
        return std::stoi(last_line.substr(0, comma));
    } catch (const std::exception&) {
        throw std::runtime_error(
            "Restart CSV has an invalid step: " + path.string());
    }
}

// A continuation appends only to the exact existing column schema.  This
// keeps every recorded field in the same column throughout the history.
// A fresh run writes the schema before its first observation.
std::ofstream open_csv_output(
    const std::filesystem::path& path,
    bool resume,
    const std::string& header,
    const char* description) {
    if (resume
        && (!std::filesystem::exists(path)
            || std::filesystem::file_size(path) == 0)) {
        throw std::runtime_error(
            std::string("Cannot resume without existing ")
            + description + ": " + path.string());
    }
    if (resume) {
        std::ifstream previous(path);
        std::string previous_header;
        std::getline(previous, previous_header);
        if (!previous_header.empty() && previous_header.back() == '\r') {
            previous_header.pop_back();
        }
        if (previous_header != header) {
            throw std::runtime_error(
                "Cannot append a changed CSV schema to existing output: " + path.string()
                + ". Start a new run directory with the new diagnostics.");
        }
    }
    std::ofstream output(
        path,
        resume ? (std::ios::out | std::ios::app)
               : (std::ios::out | std::ios::trunc));
    if (!output) {
        throw std::runtime_error(
            std::string("Could not open ") + description + ": "
            + path.string());
    }
    if (!resume) {
        output << header << '\n';
    }
    output << std::setprecision(17);
    return output;
}

void write_wpic_restart(
    const std::filesystem::path& path,
    const std::vector<ev::WeightedPICMarker>& markers,
    double time,
    int step) {
    ev::WeightedPICRestartState state;
    state.markers = markers;
    state.time = time;
    state.step = step;
    ev::write_weighted_pic_restart(path, state);
}

// These are the persistent variables for collisionless EF evolution.  The
// metric is reconstructed from the saved markers with the same fixed outer
// ADM boundary condition.  The excision mode is part of the saved model.
struct EFRestartState {
    std::vector<ev::eddington_finkelstein::Marker> markers;
    double advanced_time = 0.0;
    std::int64_t step = 0;
    double conserved_adm_mass = 0.0;
    bool enable_excision = true;
};

struct EFPostHorizonAuditState {
    double minimum_p_r = std::numeric_limits<double>::infinity();
    double minimum_b = std::numeric_limits<double>::infinity();
    double maximum_excision_radius = 0.0;
    double maximum_excised_rest_weight = 0.0;
    std::uint64_t accepted_step_samples = 0;
    bool complete = true;
    bool finite_fields_throughout = true;
    bool fixed_adm_bookkeeping_throughout = true;
    bool causal_excision_valid_throughout = true;
    bool causal_excision_inside_horizon_seen = false;
    bool excised_interior_rest_weight_seen = false;
};

std::filesystem::path ef_post_horizon_audit_path(
    const std::filesystem::path& output_dir) {
    return output_dir / "ef_post_horizon_audit.state";
}

bool finite_vector(const std::vector<double>& values) {
    return std::all_of(
        values.begin(), values.end(),
        [](double value) { return std::isfinite(value); });
}

// Accumulate accepted-history checks by logical AND: a failed condition
// remains recorded in every later summary.  Check positive future-directed
// momentum P, positive b, finite stresses, the imposed ADM boundary value,
// and the trapped buffer around any applied excision boundary.
void update_ef_post_horizon_audit(
    EFPostHorizonAuditState& audit,
    const std::vector<ev::eddington_finkelstein::Marker>& markers,
    const ev::eddington_finkelstein::Field& field,
    double conserved_adm_mass,
    bool accepted_step,
    double excision_a_threshold = -0.01,
    int excision_buffer_cells = 4) {
    const auto& metric = field.metric;
    const auto& sources = field.sources;
    bool finite = std::isfinite(field.total_rest_weight)
        && std::isfinite(metric.adm_mass)
        && std::isfinite(metric.minimum_a)
        && std::isfinite(metric.maximum_compactness)
        && std::isfinite(metric.apparent_horizon_radius)
        && std::isfinite(metric.excision_radius)
        && std::isfinite(sources.active_rest_weight)
        && std::isfinite(sources.excised_rest_weight)
        && std::isfinite(sources.escaped_rest_weight)
        && finite_vector(metric.a) && finite_vector(metric.b)
        && finite_vector(metric.a_r) && finite_vector(metric.b_r)
        && finite_vector(metric.m) && finite_vector(metric.mb)
        && finite_vector(metric.compactness)
        && finite_vector(metric.face_a) && finite_vector(metric.face_b)
        && finite_vector(metric.face_a_r) && finite_vector(metric.face_b_r)
        && finite_vector(metric.face_m) && finite_vector(metric.face_mb)
        && finite_vector(metric.face_compactness)
        && finite_vector(sources.t_rr) && finite_vector(sources.s);

    double minimum_p_r = std::numeric_limits<double>::infinity();
    for (const auto& marker : markers) {
        finite = finite
            && std::isfinite(marker.rest_weight)
            && std::isfinite(marker.r)
            && std::isfinite(marker.p_r)
            && std::isfinite(marker.eta);
        if (!(marker.p_r > 0.0)) {
            finite = false;
        }
        minimum_p_r = std::min(minimum_p_r, marker.p_r);
    }
    double minimum_b = metric.b.empty()
        ? -1.0
        : *std::min_element(metric.b.begin(), metric.b.end());
    if (!metric.face_b.empty()) {
        minimum_b = std::min(
            minimum_b,
            *std::min_element(metric.face_b.begin(), metric.face_b.end()));
    }
    finite = finite && std::isfinite(minimum_p_r)
        && minimum_p_r > 0.0 && std::isfinite(minimum_b)
        && minimum_b > 0.0;
    audit.complete = audit.complete && !markers.empty() && !metric.b.empty();
    audit.finite_fields_throughout =
        audit.finite_fields_throughout && finite;
    if (finite) {
        audit.minimum_p_r = std::min(audit.minimum_p_r, minimum_p_r);
        audit.minimum_b = std::min(audit.minimum_b, minimum_b);
        audit.maximum_excision_radius = std::max(
            audit.maximum_excision_radius, metric.excision_radius);
        audit.maximum_excised_rest_weight = std::max(
            audit.maximum_excised_rest_weight,
            sources.excised_rest_weight);
    }

    const double adm_scale = std::max(1.0, std::abs(conserved_adm_mass));
    const double adm_roundoff_tolerance = 4096.0
        * std::numeric_limits<double>::epsilon() * adm_scale;
    const bool fixed_adm_ok = metric.fixed_adm_mass_used
        && conserved_adm_mass > 0.0
        && std::isfinite(conserved_adm_mass)
        && std::abs(metric.adm_mass - conserved_adm_mass)
            <= adm_roundoff_tolerance;
    audit.fixed_adm_bookkeeping_throughout =
        audit.fixed_adm_bookkeeping_throughout && fixed_adm_ok;

    const bool causal_excision = finite
        && ev::eddington_finkelstein::causal_excision_boundary(
            metric, metric.excision_radius, excision_a_threshold,
            excision_buffer_cells);
    audit.causal_excision_valid_throughout =
        audit.causal_excision_valid_throughout
        && std::isfinite(metric.excision_radius)
        && ((metric.excision_radius == 0.0
             && sources.excised_rest_weight == 0.0) || causal_excision);
    audit.causal_excision_inside_horizon_seen =
        audit.causal_excision_inside_horizon_seen || causal_excision;
    audit.excised_interior_rest_weight_seen =
        audit.excised_interior_rest_weight_seen
        || (causal_excision && sources.excised_rest_weight > 0.0);
    if (accepted_step) {
        ++audit.accepted_step_samples;
    }
}

void copy_ef_post_horizon_audit_to_summary(
    const EFPostHorizonAuditState& audit,
    HighSummary& summary) {
    summary.ef_minimum_p_r_over_evolution =
        std::isfinite(audit.minimum_p_r) ? audit.minimum_p_r : -1.0;
    summary.ef_minimum_b_over_evolution =
        std::isfinite(audit.minimum_b) ? audit.minimum_b : -1.0;
    summary.ef_maximum_excision_radius = audit.maximum_excision_radius;
    summary.ef_maximum_excised_rest_weight =
        audit.maximum_excised_rest_weight;
    summary.ef_accepted_step_audit_samples = audit.accepted_step_samples;
    summary.ef_post_horizon_audit_complete = audit.complete;
    summary.ef_finite_fields_throughout =
        audit.finite_fields_throughout;
    summary.ef_fixed_adm_bookkeeping_throughout =
        audit.fixed_adm_bookkeeping_throughout;
    summary.ef_causal_excision_valid_throughout =
        audit.causal_excision_valid_throughout;
    summary.ef_causal_excision_inside_horizon_seen =
        audit.causal_excision_inside_horizon_seen;
    summary.ef_excised_interior_rest_weight_seen =
        audit.excised_interior_rest_weight_seen;
}

void write_ef_post_horizon_audit(
    const std::filesystem::path& path,
    const EFPostHorizonAuditState& audit,
    double advanced_time,
    std::int64_t step) {
    const std::filesystem::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Could not create EF post-horizon audit sidecar: "
            + temporary.string());
    }
    output << std::setprecision(17)
           << "VENUS_EF_POST_HORIZON_AUDIT 2\n"
           << step << ' ' << advanced_time << ' '
           << audit.minimum_p_r << ' ' << audit.minimum_b << ' '
           << audit.maximum_excision_radius << ' '
           << audit.maximum_excised_rest_weight << ' '
           << audit.accepted_step_samples << ' '
           << (audit.complete ? 1 : 0) << ' '
           << (audit.finite_fields_throughout ? 1 : 0) << ' '
           << (audit.fixed_adm_bookkeeping_throughout ? 1 : 0) << ' '
           << (audit.causal_excision_inside_horizon_seen ? 1 : 0) << ' '
           << (audit.excised_interior_rest_weight_seen ? 1 : 0) << ' '
           << (audit.causal_excision_valid_throughout ? 1 : 0) << '\n';
    output.flush();
    if (!output) {
        throw std::runtime_error(
            "Could not flush EF post-horizon audit sidecar: "
            + temporary.string());
    }
    output.close();
    std::filesystem::rename(temporary, path);
}

EFPostHorizonAuditState read_ef_post_horizon_audit(
    const std::filesystem::path& path,
    double expected_advanced_time,
    std::int64_t expected_step) {
    std::ifstream input(path);
    std::string magic;
    int version = 0;
    if (!input || !(input >> magic >> version)
        || magic != "VENUS_EF_POST_HORIZON_AUDIT"
        || (version != 1 && version != 2)) {
        throw std::runtime_error(
            "Invalid EF post-horizon audit header: " + path.string());
    }
    std::int64_t step = -1;
    double advanced_time = -1.0;
    EFPostHorizonAuditState audit;
    int complete = 0;
    int finite = 0;
    int fixed_adm = 0;
    int causal = 0;
    int excised = 0;
    if (!(input >> step >> advanced_time
          >> audit.minimum_p_r >> audit.minimum_b
          >> audit.maximum_excision_radius
          >> audit.maximum_excised_rest_weight
          >> audit.accepted_step_samples >> complete >> finite
          >> fixed_adm >> causal >> excised)
        || step != expected_step
        || advanced_time != expected_advanced_time
        || !(audit.minimum_p_r > 0.0)
        || !(audit.minimum_b > 0.0)
        || !(audit.maximum_excision_radius >= 0.0)
        || !(audit.maximum_excised_rest_weight >= 0.0)
        || !std::isfinite(audit.minimum_p_r)
        || !std::isfinite(audit.minimum_b)
        || !std::isfinite(audit.maximum_excision_radius)
        || !std::isfinite(audit.maximum_excised_rest_weight)
        || complete < 0 || complete > 1 || finite < 0 || finite > 1
        || fixed_adm < 0 || fixed_adm > 1
        || causal < 0 || causal > 1 || excised < 0 || excised > 1) {
        throw std::runtime_error(
            "Invalid or stale EF post-horizon audit state: "
            + path.string());
    }
    audit.complete = complete != 0;
    audit.finite_fields_throughout = finite != 0;
    audit.fixed_adm_bookkeeping_throughout = fixed_adm != 0;
    audit.causal_excision_inside_horizon_seen = causal != 0;
    audit.excised_interior_rest_weight_seen = excised != 0;
    if (version == 2) {
        int causal_throughout = 0;
        if (!(input >> causal_throughout)
            || causal_throughout < 0 || causal_throughout > 1) {
            throw std::runtime_error(
                "Invalid EF causal-excision audit state: " + path.string());
        }
        audit.causal_excision_valid_throughout = causal_throughout != 0;
    } else {
        // Version 1 established only r_ex<r_AH.  It cannot certify the local
        // trapped buffer on the earlier accepted slices of a resumed run.
        audit.complete = false;
        audit.causal_excision_valid_throughout = false;
        audit.causal_excision_inside_horizon_seen = false;
        audit.excised_interior_rest_weight_seen = false;
    }
    std::string trailing;
    if (input >> trailing) {
        throw std::runtime_error(
            "EF post-horizon audit sidecar contains trailing data: "
            + path.string());
    }
    return audit;
}

// Hash the exact bit patterns in their serialized order, including particle
// IDs and active/excised/escaped states.  This compact FNV fingerprint detects
// an inconsistent restart payload.  External artifact SHA-256 hashes are
// computed separately for the saved files.
std::uint64_t ef_restart_fingerprint(
    const EFRestartState& state,
    bool include_excision_mode = true) {
    constexpr std::uint64_t offset = UINT64_C(1469598103934665603);
    constexpr std::uint64_t prime = UINT64_C(1099511628211);
    std::uint64_t hash = offset;
    auto mix_u64 = [&](std::uint64_t value) {
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= (value >> (8 * byte)) & UINT64_C(0xff);
            hash *= prime;
        }
    };
    auto mix_double = [&](double value) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        mix_u64(bits);
    };
    mix_double(state.advanced_time);
    mix_u64(static_cast<std::uint64_t>(state.step));
    mix_double(state.conserved_adm_mass);
    if (include_excision_mode) {
        mix_u64(state.enable_excision ? 1 : 0);
    }
    mix_u64(static_cast<std::uint64_t>(state.markers.size()));
    for (const auto& marker : state.markers) {
        mix_double(marker.rest_weight);
        mix_double(marker.r);
        mix_double(marker.p_r);
        mix_double(marker.eta);
        mix_u64(marker.id);
        mix_u64(marker.source_id);
        mix_u64(static_cast<std::uint64_t>(marker.state));
    }
    return hash;
}

// Publish the complete particle state only after the temporary binary has
// been written and flushed.  Retaining all interior particles requires a
// state in which none have already been removed by excision.
void write_ef_restart(
    const std::filesystem::path& path,
    const EFRestartState& state) {
    if (!(state.advanced_time >= 0.0)
        || !std::isfinite(state.advanced_time)
        || state.step < 0
        || !(state.conserved_adm_mass >= 0.0)
        || !std::isfinite(state.conserved_adm_mass)) {
        throw std::runtime_error("Invalid EF restart metadata.");
    }
    if (!state.enable_excision && std::any_of(
            state.markers.begin(), state.markers.end(), [](const auto& marker) {
                return marker.state == ev::eddington_finkelstein::MarkerState::excised;
            })) {
        throw std::runtime_error(
            "EF interior evolution requires a restart with every marker retained.");
    }
    const auto nonce = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const std::filesystem::path temporary =
        path.string() + ".tmp." + std::to_string(nonce);
    try {
        std::ofstream output(
            temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "Could not create EF restart: " + temporary.string());
        }
        const char magic[16] = {
            'V','E','N','U','S','E','V','E','F','R','S','T','1',0,0,0,
        };
        output.write(magic, sizeof(magic));
        write_physical_ic_scalar<std::uint32_t>(output, 2);
        write_physical_ic_scalar(output, state.advanced_time);
        write_physical_ic_scalar(output, state.step);
        write_physical_ic_scalar(output, state.conserved_adm_mass);
        write_physical_ic_scalar<std::uint8_t>(
            output, state.enable_excision ? 1 : 0);
        write_physical_ic_scalar<std::uint64_t>(
            output, static_cast<std::uint64_t>(state.markers.size()));
        write_physical_ic_scalar(
            output, ef_restart_fingerprint(state));
        for (const auto& marker : state.markers) {
            write_physical_ic_scalar(output, marker.rest_weight);
            write_physical_ic_scalar(output, marker.r);
            write_physical_ic_scalar(output, marker.p_r);
            write_physical_ic_scalar(output, marker.eta);
            write_physical_ic_scalar(output, marker.id);
            write_physical_ic_scalar(output, marker.source_id);
            write_physical_ic_scalar(
                output, static_cast<std::uint8_t>(marker.state));
        }
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "Could not flush EF restart: " + temporary.string());
        }
        output.close();
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

EFRestartState read_ef_restart(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "Could not open EF restart: " + path.string());
    }
    char magic[16]{};
    const char expected[16] = {
        'V','E','N','U','S','E','V','E','F','R','S','T','1',0,0,0,
    };
    input.read(magic, sizeof(magic));
    const std::uint32_t version =
        read_physical_ic_scalar<std::uint32_t>(input);
    EFRestartState state;
    state.advanced_time = read_physical_ic_scalar<double>(input);
    state.step = read_physical_ic_scalar<std::int64_t>(input);
    state.conserved_adm_mass = read_physical_ic_scalar<double>(input);
    std::uint8_t excision_mode = 1;
    if (version == 2) {
        excision_mode = read_physical_ic_scalar<std::uint8_t>(input);
        state.enable_excision = excision_mode != 0;
    }
    const std::uint64_t count =
        read_physical_ic_scalar<std::uint64_t>(input);
    const std::uint64_t expected_fingerprint =
        read_physical_ic_scalar<std::uint64_t>(input);
    if (!input || std::memcmp(magic, expected, sizeof(magic)) != 0
        || (version != 1 && version != 2) || excision_mode > 1
        || !(state.advanced_time >= 0.0)
        || !std::isfinite(state.advanced_time) || state.step < 0
        || !(state.conserved_adm_mass >= 0.0)
        || !std::isfinite(state.conserved_adm_mass)
        || count > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            "Invalid EF restart header: " + path.string());
    }
    state.markers.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t q = 0; q < count; ++q) {
        ev::eddington_finkelstein::Marker marker;
        marker.rest_weight = read_physical_ic_scalar<double>(input);
        marker.r = read_physical_ic_scalar<double>(input);
        marker.p_r = read_physical_ic_scalar<double>(input);
        marker.eta = read_physical_ic_scalar<double>(input);
        marker.id = read_physical_ic_scalar<std::uint64_t>(input);
        marker.source_id = read_physical_ic_scalar<std::uint64_t>(input);
        const std::uint8_t marker_state =
            read_physical_ic_scalar<std::uint8_t>(input);
        if (!(marker.rest_weight >= 0.0)
            || !std::isfinite(marker.rest_weight)
            || !(marker.r >= 0.0) || !std::isfinite(marker.r)
            || !(marker.p_r > 0.0) || !std::isfinite(marker.p_r)
            || !(marker.eta >= 0.0) || !std::isfinite(marker.eta)
            || marker_state > static_cast<std::uint8_t>(
                ev::eddington_finkelstein::MarkerState::escaped)) {
            throw std::runtime_error(
                "Invalid marker in EF restart: " + path.string());
        }
        marker.state = static_cast<
            ev::eddington_finkelstein::MarkerState>(marker_state);
        if (!state.enable_excision
            && marker.state == ev::eddington_finkelstein::MarkerState::excised) {
            throw std::runtime_error(
                "EF interior evolution requires a restart with every marker retained.");
        }
        state.markers.push_back(marker);
    }
    char trailing = 0;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error(
            "EF restart contains trailing data: " + path.string());
    }
    if (ef_restart_fingerprint(state, version == 2) != expected_fingerprint) {
        throw std::runtime_error(
            "EF restart fingerprint mismatch: " + path.string());
    }
    return state;
}

// ---- Solved-eikonal null initial-data container --------------------------
// This container supplies particles and tensors on one solved ingoing null
// slice.  Its distinct header and manifest distinguish this initialization
// data from the EF restart that stores an evolved accepted time.
struct EFNullInitialData {
    std::uint32_t format_version = 2;
    std::vector<ev::eddington_finkelstein::Marker> markers;
    std::vector<ev::coordinate_map::MappedSlicePoint> mapped_slice;
    double conserved_adm_mass = 0.0;
    double donor_duration = 0.0;
    // outer_radius is the EF evolution boundary R_E.  The solved eikonal is
    // normalized at map_worldtube_radius R_B<=R_E; for R_B<R_E the remaining
    // cells are the exact Schwarzschild-vacuum continuation serialized below.
    double outer_radius = 0.0;
    double map_worldtube_radius = 0.0;
    double outer_advanced_time = 0.0;
    double ef_v_offset = 0.0;
    double maximum_eikonal_residual = 0.0;
    double maximum_crossing_residual = 0.0;
    double minimum_crossing_radius_clearance = 0.0;
    double minimum_worldtube_matter_clearance = 0.0;
    double maximum_worldtube_source = 0.0;
    double maximum_worldtube_b_normalization_error = 0.0;
    double worldtube_b_normalization_tolerance = 0.0;
    double maximum_analytic_vacuum_field_residual = 0.0;
    double maximum_analytic_vacuum_source = 0.0;
    double maximum_analytic_inverse_map_residual = 0.0;
    double minimum_analytic_vacuum_a = 1.0;
    std::uint64_t base_ic_fingerprint = 0;
    std::uint64_t schwarzschild_realized_fingerprint = 0;
    std::uint64_t ef_realized_fingerprint = 0;
    std::uint64_t grid_fingerprint = 0;
    std::uint64_t map_grid_fingerprint = 0;
    std::uint64_t sampling_realization = 0;
    std::uint64_t analytic_extension_start_index = 0;
    bool analytic_vacuum_extension = false;
};

class FNV64Fingerprint {
public:
    void mix_u64(std::uint64_t value) {
        for (int byte = 0; byte < 8; ++byte) {
            value_ ^= (value >> (8 * byte)) & UINT64_C(0xff);
            value_ *= UINT64_C(1099511628211);
        }
    }

    void mix_double(double value) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        mix_u64(bits);
    }

    std::uint64_t value() const { return value_; }

private:
    std::uint64_t value_ = UINT64_C(1469598103934665603);
};

std::uint64_t ef_marker_fingerprint(
    const std::vector<ev::eddington_finkelstein::Marker>& markers) {
    FNV64Fingerprint hash;
    hash.mix_u64(static_cast<std::uint64_t>(markers.size()));
    for (const auto& marker : markers) {
        hash.mix_double(marker.rest_weight);
        hash.mix_double(marker.r);
        hash.mix_double(marker.p_r);
        hash.mix_double(marker.eta);
        hash.mix_u64(marker.id);
        hash.mix_u64(marker.source_id);
        hash.mix_u64(static_cast<std::uint64_t>(marker.state));
    }
    return hash.value();
}

std::uint64_t coordinate_map_grid_prefix_fingerprint(
    const ev::WeightedPICFieldGrid& grid,
    std::size_t cell_count) {
    if (cell_count == 0 || cell_count > grid.radius.size()
        || grid.mesh.edge.size() != grid.radius.size() + 1) {
        throw std::runtime_error(
            "Invalid radial prefix for coordinate-map fingerprint.");
    }
    FNV64Fingerprint hash;
    hash.mix_u64(static_cast<std::uint64_t>(cell_count));
    for (std::size_t i = 0; i < cell_count; ++i) {
        hash.mix_double(grid.radius[i]);
    }
    for (std::size_t i = 0; i <= cell_count; ++i) {
        hash.mix_double(grid.mesh.edge[i]);
    }
    hash.mix_u64(static_cast<std::uint64_t>(
        grid.parameters.inner_boundary == "origin_parity"));
    hash.mix_u64(static_cast<std::uint64_t>(
        grid.parameters.radial_grid == "linear"));
    return hash.value();
}

std::uint64_t coordinate_map_grid_fingerprint(
    const ev::WeightedPICFieldGrid& grid) {
    return coordinate_map_grid_prefix_fingerprint(
        grid, grid.radius.size());
}

std::uint64_t null_initial_data_fingerprint(
    const EFNullInitialData& data) {
    FNV64Fingerprint hash;
    if (data.format_version >= 2) {
        hash.mix_u64(data.format_version);
    }
    hash.mix_double(data.conserved_adm_mass);
    hash.mix_double(data.donor_duration);
    hash.mix_double(data.outer_radius);
    if (data.format_version >= 2) {
        hash.mix_double(data.map_worldtube_radius);
    }
    hash.mix_double(data.outer_advanced_time);
    hash.mix_double(data.ef_v_offset);
    hash.mix_double(data.maximum_eikonal_residual);
    hash.mix_double(data.maximum_crossing_residual);
    if (data.format_version >= 2) {
        hash.mix_double(data.minimum_crossing_radius_clearance);
        hash.mix_double(data.minimum_worldtube_matter_clearance);
        hash.mix_double(data.maximum_worldtube_source);
        hash.mix_double(data.maximum_worldtube_b_normalization_error);
        hash.mix_double(data.worldtube_b_normalization_tolerance);
        hash.mix_double(data.maximum_analytic_vacuum_field_residual);
        hash.mix_double(data.maximum_analytic_vacuum_source);
        hash.mix_double(data.maximum_analytic_inverse_map_residual);
        hash.mix_double(data.minimum_analytic_vacuum_a);
    }
    hash.mix_u64(data.base_ic_fingerprint);
    hash.mix_u64(data.schwarzschild_realized_fingerprint);
    hash.mix_u64(data.ef_realized_fingerprint);
    hash.mix_u64(data.grid_fingerprint);
    if (data.format_version >= 2) {
        hash.mix_u64(data.map_grid_fingerprint);
    }
    hash.mix_u64(data.sampling_realization);
    if (data.format_version >= 2) {
        hash.mix_u64(data.analytic_extension_start_index);
        hash.mix_u64(static_cast<std::uint64_t>(
            data.analytic_vacuum_extension));
    }
    hash.mix_u64(static_cast<std::uint64_t>(data.markers.size()));
    hash.mix_u64(static_cast<std::uint64_t>(data.mapped_slice.size()));
    for (const auto& marker : data.markers) {
        hash.mix_double(marker.rest_weight);
        hash.mix_double(marker.r);
        hash.mix_double(marker.p_r);
        hash.mix_double(marker.eta);
        hash.mix_u64(marker.id);
        hash.mix_u64(marker.source_id);
        hash.mix_u64(static_cast<std::uint64_t>(marker.state));
    }
    for (const auto& point : data.mapped_slice) {
        hash.mix_double(point.r);
        hash.mix_double(point.reverse_time);
        hash.mix_double(point.schwarzschild_time);
        hash.mix_double(point.A);
        hash.mix_double(point.F);
        hash.mix_double(point.a);
        hash.mix_double(point.b);
        hash.mix_double(point.m);
        hash.mix_double(point.rho);
        hash.mix_double(point.p);
        hash.mix_double(point.pt);
        hash.mix_double(point.j);
        hash.mix_double(point.t_vv);
        hash.mix_double(point.t_vr);
        hash.mix_double(point.t_rr);
        hash.mix_double(point.t_theta_theta);
        hash.mix_double(point.s);
    }
    return hash.value();
}

void write_null_initial_data(
    const std::filesystem::path& path,
    const EFNullInitialData& data) {
    const bool extension_shape_valid = data.analytic_vacuum_extension
        ? data.analytic_extension_start_index > 0
            && data.analytic_extension_start_index
                < data.mapped_slice.size()
            && data.map_worldtube_radius < data.outer_radius
        : data.analytic_extension_start_index == data.mapped_slice.size()
            && data.map_worldtube_radius == data.outer_radius;
    if (data.format_version != 2
        || data.markers.empty() || data.mapped_slice.empty()
        || !(data.conserved_adm_mass > 0.0)
        || !(data.donor_duration > 0.0)
        || !(data.outer_radius > 0.0)
        || !(data.map_worldtube_radius > 0.0)
        || !(data.map_worldtube_radius <= data.outer_radius)
        || !extension_shape_valid
        || !(data.minimum_crossing_radius_clearance > 0.0)
        || !(data.minimum_worldtube_matter_clearance > 0.0)
        || !(data.worldtube_b_normalization_tolerance > 0.0)
        || !(data.maximum_worldtube_source >= 0.0)
        || !(data.maximum_worldtube_b_normalization_error >= 0.0)
        || data.maximum_worldtube_b_normalization_error
            > data.worldtube_b_normalization_tolerance
        || !(data.minimum_analytic_vacuum_a > 0.0)
        || !std::isfinite(data.maximum_analytic_vacuum_field_residual)
        || !std::isfinite(data.maximum_analytic_vacuum_source)
        || !std::isfinite(data.maximum_analytic_inverse_map_residual)
        || data.ef_realized_fingerprint
            != ef_marker_fingerprint(data.markers)) {
        throw std::runtime_error(
            "Refusing to write incomplete solved-eikonal initial data.");
    }
    std::filesystem::create_directories(
        path.has_parent_path() ? path.parent_path() : ".");
    const auto nonce = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const std::filesystem::path temporary =
        path.string() + ".tmp." + std::to_string(nonce);
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "Could not create null initial data: " + temporary.string());
        }
        const char magic[16] = {
            'V','E','N','U','S','N','U','L','L','M','A','P','1',0,0,0,
        };
        output.write(magic, sizeof(magic));
        write_physical_ic_scalar(output, data.format_version);
        write_physical_ic_scalar(output, data.conserved_adm_mass);
        write_physical_ic_scalar(output, data.donor_duration);
        write_physical_ic_scalar(output, data.outer_radius);
        write_physical_ic_scalar(output, data.outer_advanced_time);
        write_physical_ic_scalar(output, data.ef_v_offset);
        write_physical_ic_scalar(output, data.maximum_eikonal_residual);
        write_physical_ic_scalar(output, data.maximum_crossing_residual);
        write_physical_ic_scalar(output, data.base_ic_fingerprint);
        write_physical_ic_scalar(
            output, data.schwarzschild_realized_fingerprint);
        write_physical_ic_scalar(output, data.ef_realized_fingerprint);
        write_physical_ic_scalar(output, data.grid_fingerprint);
        write_physical_ic_scalar(output, data.sampling_realization);
        write_physical_ic_scalar(output, data.map_worldtube_radius);
        write_physical_ic_scalar(output, data.map_grid_fingerprint);
        write_physical_ic_scalar(
            output, data.analytic_extension_start_index);
        write_physical_ic_scalar<std::uint8_t>(
            output, data.analytic_vacuum_extension ? 1 : 0);
        write_physical_ic_scalar(
            output, data.minimum_crossing_radius_clearance);
        write_physical_ic_scalar(
            output, data.minimum_worldtube_matter_clearance);
        write_physical_ic_scalar(output, data.maximum_worldtube_source);
        write_physical_ic_scalar(
            output, data.maximum_worldtube_b_normalization_error);
        write_physical_ic_scalar(
            output, data.worldtube_b_normalization_tolerance);
        write_physical_ic_scalar(
            output, data.maximum_analytic_vacuum_field_residual);
        write_physical_ic_scalar(
            output, data.maximum_analytic_vacuum_source);
        write_physical_ic_scalar(
            output, data.maximum_analytic_inverse_map_residual);
        write_physical_ic_scalar(
            output, data.minimum_analytic_vacuum_a);
        write_physical_ic_scalar<std::uint64_t>(
            output, static_cast<std::uint64_t>(data.markers.size()));
        write_physical_ic_scalar<std::uint64_t>(
            output, static_cast<std::uint64_t>(data.mapped_slice.size()));
        write_physical_ic_scalar(output, null_initial_data_fingerprint(data));
        for (const auto& marker : data.markers) {
            write_physical_ic_scalar(output, marker.rest_weight);
            write_physical_ic_scalar(output, marker.r);
            write_physical_ic_scalar(output, marker.p_r);
            write_physical_ic_scalar(output, marker.eta);
            write_physical_ic_scalar(output, marker.id);
            write_physical_ic_scalar(output, marker.source_id);
            write_physical_ic_scalar(
                output, static_cast<std::uint8_t>(marker.state));
        }
        for (const auto& point : data.mapped_slice) {
            write_physical_ic_scalar(output, point.r);
            write_physical_ic_scalar(output, point.reverse_time);
            write_physical_ic_scalar(output, point.schwarzschild_time);
            write_physical_ic_scalar(output, point.A);
            write_physical_ic_scalar(output, point.F);
            write_physical_ic_scalar(output, point.a);
            write_physical_ic_scalar(output, point.b);
            write_physical_ic_scalar(output, point.m);
            write_physical_ic_scalar(output, point.rho);
            write_physical_ic_scalar(output, point.p);
            write_physical_ic_scalar(output, point.pt);
            write_physical_ic_scalar(output, point.j);
            write_physical_ic_scalar(output, point.t_vv);
            write_physical_ic_scalar(output, point.t_vr);
            write_physical_ic_scalar(output, point.t_rr);
            write_physical_ic_scalar(output, point.t_theta_theta);
            write_physical_ic_scalar(output, point.s);
        }
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "Could not flush null initial data: " + temporary.string());
        }
        output.close();
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

EFNullInitialData read_null_initial_data(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "Could not open solved-eikonal initial data: " + path.string());
    }
    char magic[16]{};
    const char expected[16] = {
        'V','E','N','U','S','N','U','L','L','M','A','P','1',0,0,0,
    };
    input.read(magic, sizeof(magic));
    const std::uint32_t version =
        read_physical_ic_scalar<std::uint32_t>(input);
    EFNullInitialData data;
    data.format_version = version;
    data.conserved_adm_mass = read_physical_ic_scalar<double>(input);
    data.donor_duration = read_physical_ic_scalar<double>(input);
    data.outer_radius = read_physical_ic_scalar<double>(input);
    data.outer_advanced_time = read_physical_ic_scalar<double>(input);
    data.ef_v_offset = read_physical_ic_scalar<double>(input);
    data.maximum_eikonal_residual = read_physical_ic_scalar<double>(input);
    data.maximum_crossing_residual = read_physical_ic_scalar<double>(input);
    data.base_ic_fingerprint = read_physical_ic_scalar<std::uint64_t>(input);
    data.schwarzschild_realized_fingerprint =
        read_physical_ic_scalar<std::uint64_t>(input);
    data.ef_realized_fingerprint =
        read_physical_ic_scalar<std::uint64_t>(input);
    data.grid_fingerprint = read_physical_ic_scalar<std::uint64_t>(input);
    data.sampling_realization =
        read_physical_ic_scalar<std::uint64_t>(input);
    std::uint8_t analytic_extension = 0;
    if (version == 2) {
        data.map_worldtube_radius =
            read_physical_ic_scalar<double>(input);
        data.map_grid_fingerprint =
            read_physical_ic_scalar<std::uint64_t>(input);
        data.analytic_extension_start_index =
            read_physical_ic_scalar<std::uint64_t>(input);
        analytic_extension = read_physical_ic_scalar<std::uint8_t>(input);
        data.analytic_vacuum_extension = analytic_extension != 0;
        data.minimum_crossing_radius_clearance =
            read_physical_ic_scalar<double>(input);
        data.minimum_worldtube_matter_clearance =
            read_physical_ic_scalar<double>(input);
        data.maximum_worldtube_source =
            read_physical_ic_scalar<double>(input);
        data.maximum_worldtube_b_normalization_error =
            read_physical_ic_scalar<double>(input);
        data.worldtube_b_normalization_tolerance =
            read_physical_ic_scalar<double>(input);
        data.maximum_analytic_vacuum_field_residual =
            read_physical_ic_scalar<double>(input);
        data.maximum_analytic_vacuum_source =
            read_physical_ic_scalar<double>(input);
        data.maximum_analytic_inverse_map_residual =
            read_physical_ic_scalar<double>(input);
        data.minimum_analytic_vacuum_a =
            read_physical_ic_scalar<double>(input);
    } else {
        data.map_worldtube_radius = data.outer_radius;
        data.map_grid_fingerprint = data.grid_fingerprint;
    }
    const std::uint64_t marker_count =
        read_physical_ic_scalar<std::uint64_t>(input);
    const std::uint64_t slice_count =
        read_physical_ic_scalar<std::uint64_t>(input);
    if (version == 1) {
        data.analytic_extension_start_index = slice_count;
    }
    const std::uint64_t expected_fingerprint =
        read_physical_ic_scalar<std::uint64_t>(input);
    const std::uint64_t maximum_size = static_cast<std::uint64_t>(
        std::numeric_limits<std::size_t>::max());
    if (!input || std::memcmp(magic, expected, sizeof(magic)) != 0
        || (version != 1 && version != 2)
        || marker_count == 0 || slice_count == 0
        || marker_count > maximum_size || slice_count > maximum_size
        || !(data.conserved_adm_mass > 0.0)
        || !(data.donor_duration > 0.0)
        || !(data.outer_radius > 0.0)
        || !std::isfinite(data.maximum_eikonal_residual)
        || !std::isfinite(data.maximum_crossing_residual)
        || (version == 2
            && (!(data.map_worldtube_radius > 0.0)
                || !(data.map_worldtube_radius <= data.outer_radius)
                || analytic_extension > 1
                || !(data.minimum_crossing_radius_clearance > 0.0)
                || !(data.minimum_worldtube_matter_clearance > 0.0)
                || !(data.maximum_worldtube_source >= 0.0)
                || !(data.maximum_worldtube_b_normalization_error >= 0.0)
                || !(data.worldtube_b_normalization_tolerance > 0.0)
                || data.maximum_worldtube_b_normalization_error
                    > data.worldtube_b_normalization_tolerance
                || !std::isfinite(data.maximum_worldtube_source)
                || !std::isfinite(
                    data.maximum_worldtube_b_normalization_error)
                || !std::isfinite(
                    data.worldtube_b_normalization_tolerance)
                || !(data.minimum_analytic_vacuum_a > 0.0)
                || !std::isfinite(
                    data.maximum_analytic_vacuum_field_residual)
                || !std::isfinite(data.maximum_analytic_vacuum_source)
                || !std::isfinite(
                    data.maximum_analytic_inverse_map_residual)
                || (data.analytic_vacuum_extension
                    ? data.analytic_extension_start_index == 0
                        || data.analytic_extension_start_index >= slice_count
                        || !(data.map_worldtube_radius < data.outer_radius)
                    : data.analytic_extension_start_index != slice_count
                        || data.map_worldtube_radius
                            != data.outer_radius)))) {
        throw std::runtime_error(
            "Invalid solved-eikonal initial-data header: " + path.string());
    }
    data.markers.reserve(static_cast<std::size_t>(marker_count));
    for (std::uint64_t q = 0; q < marker_count; ++q) {
        ev::eddington_finkelstein::Marker marker;
        marker.rest_weight = read_physical_ic_scalar<double>(input);
        marker.r = read_physical_ic_scalar<double>(input);
        marker.p_r = read_physical_ic_scalar<double>(input);
        marker.eta = read_physical_ic_scalar<double>(input);
        marker.id = read_physical_ic_scalar<std::uint64_t>(input);
        marker.source_id = read_physical_ic_scalar<std::uint64_t>(input);
        const std::uint8_t state = read_physical_ic_scalar<std::uint8_t>(input);
        if (!(marker.rest_weight >= 0.0) || !(marker.r > 0.0)
            || !(marker.p_r > 0.0) || !(marker.eta >= 0.0)
            || !std::isfinite(marker.rest_weight)
            || !std::isfinite(marker.r) || !std::isfinite(marker.p_r)
            || !std::isfinite(marker.eta)
            || state != static_cast<std::uint8_t>(
                ev::eddington_finkelstein::MarkerState::active)) {
            throw std::runtime_error(
                "Invalid marker in solved-eikonal initial data.");
        }
        marker.state = ev::eddington_finkelstein::MarkerState::active;
        data.markers.push_back(marker);
    }
    data.mapped_slice.reserve(static_cast<std::size_t>(slice_count));
    for (std::uint64_t i = 0; i < slice_count; ++i) {
        ev::coordinate_map::MappedSlicePoint point;
        point.r = read_physical_ic_scalar<double>(input);
        point.reverse_time = read_physical_ic_scalar<double>(input);
        point.schwarzschild_time = read_physical_ic_scalar<double>(input);
        point.A = read_physical_ic_scalar<double>(input);
        point.F = read_physical_ic_scalar<double>(input);
        point.a = read_physical_ic_scalar<double>(input);
        point.b = read_physical_ic_scalar<double>(input);
        point.m = read_physical_ic_scalar<double>(input);
        point.rho = read_physical_ic_scalar<double>(input);
        point.p = read_physical_ic_scalar<double>(input);
        point.pt = read_physical_ic_scalar<double>(input);
        point.j = read_physical_ic_scalar<double>(input);
        point.t_vv = read_physical_ic_scalar<double>(input);
        point.t_vr = read_physical_ic_scalar<double>(input);
        point.t_rr = read_physical_ic_scalar<double>(input);
        point.t_theta_theta = read_physical_ic_scalar<double>(input);
        point.s = read_physical_ic_scalar<double>(input);
        if (!(point.r > 0.0) || !(point.A > 0.0)
            || !(point.F > 0.0) || !(point.a > 0.0)
            || !(point.b > 0.0) || !std::isfinite(point.s)
            || !std::isfinite(point.reverse_time)
            || !std::isfinite(point.schwarzschild_time)
            || !std::isfinite(point.m) || !std::isfinite(point.rho)
            || !std::isfinite(point.p) || !std::isfinite(point.pt)
            || !std::isfinite(point.j) || !std::isfinite(point.t_vv)
            || !std::isfinite(point.t_vr) || !std::isfinite(point.t_rr)
            || !std::isfinite(point.t_theta_theta)
            || (!data.mapped_slice.empty()
                && !(point.r > data.mapped_slice.back().r))) {
            throw std::runtime_error(
                "Invalid mapped tensor point in null initial data.");
        }
        data.mapped_slice.push_back(point);
    }
    if (version == 2) {
        const std::size_t start = static_cast<std::size_t>(
            data.analytic_extension_start_index);
        const double radius_tolerance = 128.0
            * std::numeric_limits<double>::epsilon()
            * std::max(1.0, data.outer_radius);
        if (!(data.mapped_slice.back().r < data.outer_radius)
            || std::abs(data.outer_advanced_time + data.donor_duration)
                > radius_tolerance
            || std::abs(data.ef_v_offset - data.donor_duration)
                > radius_tolerance) {
            throw std::runtime_error(
                "Null-map v2 time/domain metadata are inconsistent.");
        }
        for (std::size_t i = 0; i < data.mapped_slice.size(); ++i) {
            const bool numerical = i < start;
            if ((numerical
                    && !(data.mapped_slice[i].r
                        < data.map_worldtube_radius))
                || (!numerical
                    && !(data.mapped_slice[i].r
                        > data.map_worldtube_radius))) {
                throw std::runtime_error(
                    "Null-map v2 extension index does not bracket R_B.");
            }
        }
        if (data.analytic_vacuum_extension) {
            const auto audit =
                ev::coordinate_map::audit_schwarzschild_vacuum_extension(
                    data.mapped_slice, start,
                    data.map_worldtube_radius, data.donor_duration,
                    data.conserved_adm_mass);
            const double audit_tolerance = 256.0
                * std::numeric_limits<double>::epsilon()
                * std::max({1.0, data.outer_radius,
                    data.donor_duration, data.conserved_adm_mass});
            if (std::abs(audit.maximum_field_residual
                    - data.maximum_analytic_vacuum_field_residual)
                    > audit_tolerance
                || std::abs(audit.maximum_source
                    - data.maximum_analytic_vacuum_source)
                    > audit_tolerance
                || std::abs(audit.maximum_inverse_map_residual
                    - data.maximum_analytic_inverse_map_residual)
                    > audit_tolerance
                || std::abs(audit.minimum_a
                    - data.minimum_analytic_vacuum_a)
                    > audit_tolerance) {
                throw std::runtime_error(
                    "Null-map v2 analytic-vacuum audit does not reproduce.");
            }
        } else if (data.maximum_analytic_vacuum_field_residual != 0.0
            || data.maximum_analytic_vacuum_source != 0.0
            || data.maximum_analytic_inverse_map_residual != 0.0
            || data.minimum_analytic_vacuum_a != 1.0) {
            throw std::runtime_error(
                "Unextended null-map v2 data carry spurious vacuum audit.");
        }
    }
    char trailing = 0;
    if (input.read(&trailing, 1)
        || ef_marker_fingerprint(data.markers)
            != data.ef_realized_fingerprint
        || null_initial_data_fingerprint(data) != expected_fingerprint) {
        throw std::runtime_error(
            "Solved-eikonal initial-data integrity check failed: "
            + path.string());
    }
    return data;
}

std::filesystem::path null_artifact_path(
    const std::filesystem::path& base,
    const char* suffix) {
    return std::filesystem::path(base.string() + suffix);
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open manifest: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("Could not read manifest: " + path.string());
    }
    return buffer.str();
}

std::string json_string_field(
    const std::string& document,
    const std::string& key) {
    const std::string token = "\"" + key + "\"";
    std::size_t position = document.find(token);
    if (position == std::string::npos
        || (position = document.find(':', position + token.size()))
            == std::string::npos
        || (position = document.find('"', position + 1))
            == std::string::npos) {
        throw std::runtime_error("Manifest is missing string key " + key);
    }
    const std::size_t end = document.find('"', position + 1);
    if (end == std::string::npos) {
        throw std::runtime_error("Manifest has malformed string key " + key);
    }
    return document.substr(position + 1, end - position - 1);
}

double json_number_field(
    const std::string& document,
    const std::string& key) {
    const std::string token = "\"" + key + "\"";
    std::size_t position = document.find(token);
    if (position == std::string::npos
        || (position = document.find(':', position + token.size()))
            == std::string::npos) {
        throw std::runtime_error("Manifest is missing numeric key " + key);
    }
    try {
        std::size_t consumed = 0;
        const double value = std::stod(document.substr(position + 1),
                                       &consumed);
        if (consumed == 0 || !std::isfinite(value)) {
            throw std::runtime_error("non-finite");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("Manifest has malformed numeric key " + key);
    }
}

bool json_boolean_field(
    const std::string& document,
    const std::string& key) {
    const std::string token = "\"" + key + "\"";
    std::size_t position = document.find(token);
    if (position == std::string::npos
        || (position = document.find(':', position + token.size()))
            == std::string::npos) {
        throw std::runtime_error("Manifest is missing Boolean key " + key);
    }
    const std::size_t value = document.find_first_not_of(
        " \t\r\n", position + 1);
    if (value != std::string::npos && document.compare(value, 4, "true") == 0) {
        return true;
    }
    if (value != std::string::npos && document.compare(value, 5, "false") == 0) {
        return false;
    }
    throw std::runtime_error("Manifest has malformed Boolean key " + key);
}

std::string json_artifact_sha256(
    const std::string& document,
    const std::string& artifact_name) {
    const std::string token = "\"" + artifact_name + "\"";
    const std::size_t begin = document.find(token);
    if (begin == std::string::npos) {
        throw std::runtime_error(
            "Manifest is missing artifact " + artifact_name);
    }
    const std::size_t object_end = document.find('}', begin + token.size());
    if (object_end == std::string::npos) {
        throw std::runtime_error(
            "Manifest has malformed artifact " + artifact_name);
    }
    return json_string_field(
        document.substr(begin, object_end - begin + 1), "sha256");
}

// The manifest joins particles, tensor slice, trajectory crossings and the
// donor history. Verify every SHA-256 so these files describe one extraction.
void verify_null_manifest_artifacts(const std::filesystem::path& base) {
    const std::filesystem::path manifest =
        null_artifact_path(base, ".manifest.json");
    const std::string document = read_text_file(manifest);
    if (document.find("\"complete\": true") == std::string::npos
        || document.find("\"all_markers_crossed_once\": true")
            == std::string::npos
        || document.find("\"outer_worldtube_verified_vacuum\": true")
            == std::string::npos) {
        throw std::runtime_error(
            "Null-map manifest is incomplete or lacks required proofs.");
    }
    const std::pair<const char*, std::filesystem::path> artifact[] = {
        {"null_initial_data", base},
        {"crossings", null_artifact_path(base, ".crossings.csv")},
        {"slice", null_artifact_path(base, ".slice.csv")},
        {"accepted_steps",
            null_artifact_path(base, ".accepted_steps.csv")},
        {"donor_history",
            null_artifact_path(base, ".donor_history.bin")},
    };
    for (const auto& item : artifact) {
        const std::string expected = json_artifact_sha256(
            document, item.first);
        const std::string actual = ev::integrity::sha256_file(item.second);
        if (expected != actual) {
            throw std::runtime_error(
                std::string("Null-map artifact SHA-256 mismatch for ")
                + item.second.string());
        }
    }
}

std::string hexadecimal_fingerprint(std::uint64_t value) {
    std::ostringstream output;
    output << "0x" << std::hex << std::setw(16) << std::setfill('0')
           << value;
    return output.str();
}

void verify_null_manifest_metadata(
    const std::filesystem::path& base,
    const EFNullInitialData& data) {
    const std::string document = read_text_file(
        null_artifact_path(base, ".manifest.json"));
    const std::pair<const char*, std::uint64_t> fingerprint[] = {
        {"base_ic_fingerprint", data.base_ic_fingerprint},
        {"schwarzschild_realized_fingerprint",
            data.schwarzschild_realized_fingerprint},
        {"ef_realized_fingerprint", data.ef_realized_fingerprint},
        {"grid_fingerprint", data.grid_fingerprint},
        {"null_map_fingerprint", null_initial_data_fingerprint(data)},
    };
    for (const auto& item : fingerprint) {
        if (json_string_field(document, item.first)
            != hexadecimal_fingerprint(item.second)) {
            throw std::runtime_error(
                std::string("Null-map manifest metadata mismatch for ")
                + item.first);
        }
    }
    const std::string realization_token =
        "\"sampling_realization\": "
        + std::to_string(data.sampling_realization);
    const std::string count_token = "\"marker_count\": "
        + std::to_string(data.markers.size());
    if (document.find(realization_token) == std::string::npos
        || document.find(count_token) == std::string::npos) {
        throw std::runtime_error(
            "Null-map manifest realization/count metadata mismatch.");
    }
    const double schema_version = json_number_field(
        document, "schema_version");
    if (schema_version != static_cast<double>(data.format_version)) {
        throw std::runtime_error(
            "Null-map manifest/binary format-version mismatch.");
    }
    if (data.format_version == 2) {
        if (json_string_field(document, "map_grid_fingerprint")
                != hexadecimal_fingerprint(data.map_grid_fingerprint)
            || json_boolean_field(document, "analytic_vacuum_extension")
                != data.analytic_vacuum_extension
            || json_number_field(document,
                    "analytic_vacuum_extension_start_index")
                != static_cast<double>(
                    data.analytic_extension_start_index)
            || json_number_field(document, "map_worldtube_radius")
                != data.map_worldtube_radius
            || json_number_field(document, "evolution_outer_radius")
                != data.outer_radius
            || json_number_field(document,
                    "minimum_crossing_radius_clearance")
                != data.minimum_crossing_radius_clearance
            || json_number_field(document,
                    "minimum_worldtube_matter_clearance")
                != data.minimum_worldtube_matter_clearance
            || json_number_field(document, "maximum_outer_source")
                != data.maximum_worldtube_source
            || json_number_field(document,
                    "maximum_outer_b_normalization_error")
                != data.maximum_worldtube_b_normalization_error
            || json_number_field(document,
                    "outer_b_normalization_tolerance")
                != data.worldtube_b_normalization_tolerance
            || json_number_field(document,
                    "maximum_analytic_vacuum_field_residual")
                != data.maximum_analytic_vacuum_field_residual
            || json_number_field(document,
                    "maximum_analytic_vacuum_source")
                != data.maximum_analytic_vacuum_source
            || json_number_field(document,
                    "maximum_analytic_inverse_map_residual")
                != data.maximum_analytic_inverse_map_residual) {
            throw std::runtime_error(
                "Null-map v2 manifest extension metadata mismatch.");
        }
    }
}

std::size_t null_map_worldtube_cell_count(
    const ev::WeightedPICFieldGrid& grid,
    double map_worldtube_radius) {
    if (grid.mesh.edge.size() != grid.radius.size() + 1) {
        throw std::runtime_error("Null-map grid lacks a complete edge mesh.");
    }
    const double tolerance = 128.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(grid.mesh.edge.back()));
    const auto edge = std::lower_bound(
        grid.mesh.edge.begin(), grid.mesh.edge.end(),
        map_worldtube_radius - tolerance);
    if (edge == grid.mesh.edge.end()
        || std::abs(*edge - map_worldtube_radius) > tolerance
        || edge == grid.mesh.edge.begin()) {
        throw std::runtime_error(
            "Null-map worldtube is not an exact non-inner mesh edge.");
    }
    return static_cast<std::size_t>(edge - grid.mesh.edge.begin());
}

// The EF constraint solve must use the same radial cells as the mapped
// tensor data. Check cell faces and vacuum clearance before using that slice.
// An outer analytic extension is checked against the isolated vacuum metric.
void validate_null_initial_data_for_grid(
    const EFNullInitialData& data,
    const ev::WeightedPICFieldGrid& grid) {
    const double tolerance = 128.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(grid.mesh.edge.back()));
    if (data.mapped_slice.size() != grid.radius.size()
        || data.grid_fingerprint != coordinate_map_grid_fingerprint(grid)
        || std::abs(data.outer_radius - grid.mesh.edge.back()) > tolerance) {
        throw std::runtime_error(
            "Solved-eikonal data do not match the full EF evolution grid.");
    }
    for (std::size_t i = 0; i < grid.radius.size(); ++i) {
        if (std::abs(data.mapped_slice[i].r - grid.radius[i]) > tolerance) {
            throw std::runtime_error(
                "Solved-eikonal tensor slice is not on the EF grid.");
        }
    }
    if (data.format_version == 1) {
        return;
    }
    if (data.format_version != 2) {
        throw std::runtime_error("Unsupported null-map format version.");
    }
    const std::size_t map_cells = null_map_worldtube_cell_count(
        grid, data.map_worldtube_radius);
    const bool split = map_cells < grid.radius.size();
    const double local_dr = grid.mesh.edge[map_cells]
        - grid.mesh.edge[map_cells - 1];
    const double required_clearance = 4.0 * local_dr;
    if (data.map_grid_fingerprint
            != coordinate_map_grid_prefix_fingerprint(grid, map_cells)
        || data.analytic_extension_start_index != map_cells
        || data.analytic_vacuum_extension != split
        || data.minimum_crossing_radius_clearance
            < required_clearance - tolerance
        || data.minimum_worldtube_matter_clearance
            < required_clearance - tolerance
        || data.maximum_worldtube_source > 1.0e-12
        || data.maximum_worldtube_b_normalization_error
            > data.worldtube_b_normalization_tolerance
        || data.worldtube_b_normalization_tolerance > 5.0e-4) {
        throw std::runtime_error(
            "Null-map v2 prefix fingerprint, extension index, worldtube "
            "vacuum normalization, or matter-clearance proof is invalid "
            "for the EF grid.");
    }
    if (split) {
        const auto audit =
            ev::coordinate_map::audit_schwarzschild_vacuum_extension(
                data.mapped_slice, map_cells,
                data.map_worldtube_radius, data.donor_duration,
                data.conserved_adm_mass);
        const double residual_tolerance = 2.0e-12
            * std::max({1.0, data.outer_radius, data.donor_duration});
        if (audit.maximum_field_residual > residual_tolerance
            || audit.maximum_source > residual_tolerance
            || audit.maximum_inverse_map_residual > residual_tolerance
            || data.maximum_analytic_vacuum_field_residual
                > residual_tolerance
            || data.maximum_analytic_vacuum_source > residual_tolerance
            || data.maximum_analytic_inverse_map_residual
                > residual_tolerance) {
            throw std::runtime_error(
                "Null-map analytic Schwarzschild-vacuum continuation "
                "failed its exact-field/source/inverse-map audit.");
        }
    }
}

struct LowRestartPair {
    ev::WeightedPICRestartState einstein_vlasov;
    ev::WeightedPICRestartState vlasov_poisson;
};

struct LowRestartManifest {
    char slot = 'A';
    double time = 0.0;
    std::int64_t step = 0;
};

std::filesystem::path low_restart_manifest_path(
    const std::filesystem::path& output_dir) {
    return output_dir / "low_restart_manifest.txt";
}

std::filesystem::path low_restart_state_path(
    const std::filesystem::path& output_dir,
    const char* model,
    char slot) {
    return output_dir
        / (std::string("low_") + model + "_" + slot + ".restart");
}

LowRestartManifest read_low_restart_manifest(
    const std::filesystem::path& output_dir) {
    const std::filesystem::path path =
        low_restart_manifest_path(output_dir);
    std::ifstream input(path);
    std::string magic;
    int version = 0;
    LowRestartManifest result;
    if (!(input >> magic >> version >> result.slot
          >> result.time >> result.step)
        || magic != "VENUS_LOW_EV_VP_RESTART" || version != 1
        || !(result.slot == 'A' || result.slot == 'B')
        || !(result.time >= 0.0) || !std::isfinite(result.time)
        || result.step < 0) {
        throw std::runtime_error(
            "Invalid low EV/VP restart manifest: " + path.string());
    }
    input >> std::ws;
    if (!input.eof()) {
        throw std::runtime_error(
            "Low EV/VP restart manifest has trailing data: "
            + path.string());
    }
    return result;
}

LowRestartPair read_low_restart_pair(
    const std::filesystem::path& output_dir) {
    const LowRestartManifest manifest =
        read_low_restart_manifest(output_dir);
    LowRestartPair result;
    result.einstein_vlasov = ev::read_weighted_pic_restart(
        low_restart_state_path(
            output_dir, "einstein_vlasov", manifest.slot));
    result.vlasov_poisson = ev::read_weighted_pic_restart(
        low_restart_state_path(
            output_dir, "vlasov_poisson", manifest.slot));
    if (result.einstein_vlasov.step != manifest.step
        || result.vlasov_poisson.step != manifest.step
        || result.einstein_vlasov.time != manifest.time
        || result.vlasov_poisson.time != manifest.time) {
        throw std::runtime_error(
            "Low EV/VP particle checkpoints do not match their manifest.");
    }
    return result;
}

// Write both models into the inactive A/B slot. Publish the small manifest
// only after both complete states exist at the same step and time. A restart
// therefore selects a complete EV/VP comparison even if writing was interrupted.
void write_low_restart_pair(
    const std::filesystem::path& output_dir,
    const std::vector<ev::WeightedPICMarker>& ev_markers,
    const std::vector<ev::WeightedPICMarker>& vp_markers,
    double time,
    int step) {
    char slot = 'A';
    const std::filesystem::path manifest_path =
        low_restart_manifest_path(output_dir);
    if (std::filesystem::exists(manifest_path)) {
        const char active = read_low_restart_manifest(output_dir).slot;
        slot = active == 'A' ? 'B' : 'A';
    }
    write_wpic_restart(
        low_restart_state_path(
            output_dir, "einstein_vlasov", slot),
        ev_markers,
        time,
        step);
    write_wpic_restart(
        low_restart_state_path(
            output_dir, "vlasov_poisson", slot),
        vp_markers,
        time,
        step);

    const std::filesystem::path temporary =
        manifest_path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Could not create low EV/VP restart manifest.");
    }
    output << std::setprecision(17)
           << "VENUS_LOW_EV_VP_RESTART 1 " << slot << ' '
           << time << ' ' << step << '\n';
    output.flush();
    if (!output) {
        throw std::runtime_error(
            "Failed to flush low EV/VP restart manifest.");
    }
    output.close();
    if (std::rename(temporary.c_str(), manifest_path.c_str()) != 0) {
        throw std::runtime_error(
            "Could not install low EV/VP restart manifest.");
    }
}

struct LowResumeHistory {
    LowSummary summary;
    double initial_rest_mass = 0.0;
    double initial_vp_mass = 0.0;
    double initial_adm_mass = 0.0;
    double minimum_r50 = std::numeric_limits<double>::infinity();
};

// Continue extrema and initial reference masses from the existing history.
// The reference remains the beginning of the physical run after a restart.
LowResumeHistory read_low_resume_history(
    const std::filesystem::path& path) {
    const auto rows = read_numeric_csv_history(path, 19);
    LowResumeHistory result;
    const auto& first = rows.front();
    result.initial_rest_mass = first[3];
    result.initial_adm_mass = first[4];
    result.initial_vp_mass = first[11];
    result.summary.initial_compactness = first[5];
    result.summary.maximum_compactness = first[5];
    result.summary.maximum_compactness_time = first[1];
    result.summary.minimum_metric_gap = first[6];
    result.summary.initial_virial_ratio = first[10];
    result.summary.initial_r50 = first[8];
    result.minimum_r50 = first[8];
    for (const auto& row : rows) {
        if (row[5] > result.summary.maximum_compactness) {
            result.summary.maximum_compactness = row[5];
            result.summary.maximum_compactness_time = row[1];
        }
        result.summary.minimum_metric_gap = std::min(
            result.summary.minimum_metric_gap, row[6]);
        result.summary.maximum_adm_mass_relative_drift = std::max(
            result.summary.maximum_adm_mass_relative_drift,
            std::abs(row[4] - result.initial_adm_mass)
                / result.initial_adm_mass);
        result.minimum_r50 = std::min(result.minimum_r50, row[8]);
    }
    return result;
}

struct HighResumeHistory {
    HighSummary summary;
    double initial_rest_mass = 0.0;
};

HighResumeHistory read_high_resume_history(
    const std::filesystem::path& path) {
    const auto rows = read_numeric_csv_history(path, 14);
    const auto& first = rows.front();
    HighResumeHistory result;
    result.initial_rest_mass = first[3];
    result.summary.initial_adm_mass = first[4];
    result.summary.initial_deposited_adm_mass = first[4];
    result.summary.initial_compactness = first[5];
    result.summary.initial_min_gap = first[6];
    result.summary.initial_min_lapse = first[7];
    result.summary.initial_r50 = first[10];
    result.summary.minimum_r50 = first[10];
    for (const auto& row : rows) {
        result.summary.minimum_r50 = std::min(
            result.summary.minimum_r50, row[10]);
        result.summary.final_anisotropic_jeans_relative_l1 = row[13];
    }
    return result;
}

HighResumeHistory read_ef_high_resume_history(
    const std::filesystem::path& path) {
    const auto rows = read_numeric_csv_history(path, 21);
    const auto& first = rows.front();
    HighResumeHistory result;
    result.initial_rest_mass = first[3];
    result.summary.initial_adm_mass = first[7];
    result.summary.initial_deposited_adm_mass = first[7];
    result.summary.initial_compactness = first[8];
    result.summary.initial_min_gap = first[9];
    result.summary.initial_min_b = first[10];
    result.summary.initial_r50 = first[13];
    result.summary.minimum_r50 = first[13];
    for (const auto& row : rows) {
        result.summary.minimum_r50 = std::min(
            result.summary.minimum_r50, row[13]);
        if (row[11] >= 0.0 && result.summary.first_trapped_time < 0.0) {
            result.summary.first_trapped_time = row[1];
            result.summary.horizon_time = row[1];
        }
        result.summary.excised_rest_mass = row[5];
        result.summary.ef_last_accepted_dt = row[2];
        result.summary.escaped_rest_mass = row[6];
        result.summary.apparent_horizon_radius = row[11];
        result.summary.final_anisotropic_jeans_relative_l1 = row[17];
    }
    result.summary.horizon_formed =
        result.summary.first_trapped_time >= 0.0;
    return result;
}

std::vector<ev::EinsteinVlasovHaloHistorySample>
read_high_halo_classifier_history(
    const std::filesystem::path& path,
    bool eddington_finkelstein) {
    const std::size_t columns = eddington_finkelstein ? 21 : 14;
    const auto rows = read_numeric_csv_history(path, columns);
    std::vector<ev::EinsteinVlasovHaloHistorySample> result;
    result.reserve(rows.size());
    for (const auto& row : rows) {
        ev::EinsteinVlasovHaloHistorySample sample;
        sample.time = row[1];
        sample.maximum_compactness =
            row[eddington_finkelstein ? 8 : 5];
        sample.r50 = row[eddington_finkelstein ? 13 : 10];
        sample.r90 = row[eddington_finkelstein ? 14 : 11];
        sample.anisotropic_jeans_relative_l1 =
            row[eddington_finkelstein ? 17 : 13];
        result.push_back(sample);
    }
    return result;
}

double escaped_schwarzschild_rest_weight(
    const std::vector<ev::WeightedPICMarker>& markers) {
    long double result = 0.0L;
    for (const auto& marker : markers) {
        if (marker.state == ev::WeightedPICMarkerState::escaped) {
            result += marker.mass;
        }
    }
    return static_cast<double>(result);
}

ev::EinsteinVlasovHaloClassification classify_high_halo_history(
    const std::filesystem::path& history_path,
    bool eddington_finkelstein,
    const HighSummary& summary,
    double configured_end_time,
    double initial_rest_weight) {
    ev::EinsteinVlasovHaloClassifierInput input;
    input.fixed_family_binding_diagnostics_available =
        summary.fixed_family_binding_diagnostics_available;
    input.fixed_family_binding_fraction =
        summary.fixed_family_binding_fraction;
    input.horizon_formed = summary.horizon_formed;
    input.configured_end_time = configured_end_time;
    input.initial_rest_weight = initial_rest_weight;
    input.escaped_rest_weight = summary.escaped_rest_mass;
    return ev::classify_einstein_vlasov_halo(
        read_high_halo_classifier_history(
            history_path, eddington_finkelstein),
        input);
}

// ---- Weak-field Einstein--Vlasov versus Vlasov--Poisson experiment -------
// Compare relativistic and Newtonian gravity for the same sampled positions
// and momenta in a weak field.  EV weights use the invariant relativistic
// measure, while VP keeps coordinate-volume masses.  Both populations advance
// through the same accepted intervals so their profiles can be compared.
LowSummary run_low_compactness(const RunOptions& options) {
    std::cout << "Running low-compactness EV/VP halo comparison with "
              << options.low_markers << " markers...\n";
    ev::Params ev_parameters =
        base_parameters(options.low_radial_cells, 0.015, 20.0);
    ev_parameters.gap_stop = 0.02;
    ev_parameters.vp_timestep_omitted_mass_fraction =
        options.timestep_omitted_mass_fraction;
    const ev::WeightedPICFieldGrid grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(
            ev_parameters, ev_parameters.Nr);
    ev::Params vp_parameters = grid.parameters;
    vp_parameters.physics_model = "vlasov_poisson";
    // The VP KDK comparison requires a closed numerical outer face.
    // The Plummer support is kept far from that face, while the EV evolution
    // itself uses the isolated outflow boundary.
    vp_parameters.outer_boundary = "reflecting";

    const std::filesystem::path history_path =
        options.output_dir / "low_compactness_history.csv";
    const std::filesystem::path profiles_path =
        options.output_dir / "low_compactness_profiles.csv";

    std::vector<ev::WeightedPICMarker> ev_markers;
    std::vector<ev::WeightedPICMarker> vp_markers;
    LowSummary summary;
    double time = 0.0;
    int step = 0;
    double initial_rest_mass = 0.0;
    double initial_vp_mass = 0.0;
    double initial_adm_mass = 0.0;
    double minimum_r50 = std::numeric_limits<double>::infinity();
    if (options.resume) {
        const LowRestartPair restart =
            read_low_restart_pair(options.output_dir);
        const ev::WeightedPICRestartState& ev_restart =
            restart.einstein_vlasov;
        const ev::WeightedPICRestartState& vp_restart =
            restart.vlasov_poisson;
        if (ev_restart.step != vp_restart.step
            || ev_restart.time != vp_restart.time
            || ev_restart.step > std::numeric_limits<int>::max()
            || ev_restart.markers.size() != options.low_markers
            || vp_restart.markers.size() != options.low_markers) {
            throw std::runtime_error(
                "Low EV/VP restart states or configured marker count "
                "do not match.");
        }
        ev_markers = ev_restart.markers;
        vp_markers = vp_restart.markers;
        time = ev_restart.time;
        step = static_cast<int>(ev_restart.step);
        const LowResumeHistory accumulated =
            read_low_resume_history(history_path);
        summary = accumulated.summary;
        initial_rest_mass = accumulated.initial_rest_mass;
        initial_vp_mass = accumulated.initial_vp_mass;
        initial_adm_mass = accumulated.initial_adm_mass;
        minimum_r50 = accumulated.minimum_r50;
        if (last_csv_step(history_path) > step
            || last_csv_step(profiles_path) > step) {
            throw std::runtime_error(
                "Low restart checkpoint is older than existing CSV output.");
        }
        std::cout << "  resumed low EV/VP checkpoint at step=" << step
                  << " t=" << std::setprecision(17) << time << '\n';
    } else {
        ev_markers = make_cold_plummer_markers(
            options.low_markers,
            options.low_total_mass,
            1.0,
            ev_parameters.r_min,
            ev_parameters.r_max,
            options.low_velocity_factor,
            options.low_radial_importance_exponent,
            0);
        vp_markers = ev_markers;
        // The Plummer sampler supplies coordinate-volume masses.  EV marker
        // masses live in the invariant e^lambda dr dw d(eta) measure, whereas
        // the VP comparison keeps the original coordinate weights.
        ev::canonicalize_weighted_einstein_vlasov_pic_initial_weights(
            ev_markers, ev_parameters);
        initial_rest_mass =
            ev::weighted_pic_total_marker_mass(ev_markers);
        initial_vp_mass =
            ev::weighted_pic_total_marker_mass(vp_markers);
    }
    ev::WeightedEinsteinVlasovPICOptions step_options;
    step_options.shape = ev::WeightedPICParticleShape::tsc;
    step_options.threads = options.threads;
    step_options.characteristic_substeps = 1;
    step_options.require_vacuum_outer_boundary = true;
    ev::check_weighted_einstein_vlasov_pic_vacuum_outer_boundary(ev_markers);

    std::ofstream history = open_csv_output(
        history_path,
        options.resume,
        "step,time,dt,ev_rest_mass,ev_adm_mass,ev_max_compactness,"
           "ev_min_gap,ev_r10,ev_r50,ev_r90,ev_virial_ratio,"
           "vp_mass,vp_r10,vp_r50,vp_r90,vp_virial_ratio,"
           "density_l1_difference,particle_rms_radius_difference,"
           "particle_rms_momentum_difference",
        "low_compactness_history.csv");
    std::ofstream profiles = open_csv_output(
        profiles_path,
        options.resume,
        "step,time,r,ev_rho,ev_compactness,vp_rho,vp_compactness",
        "low_compactness_profiles.csv");

    ev::WeightedEinsteinVlasovPICField ev_field =
        ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
            ev_markers, grid, step_options.shape, options.threads);
    auto vp_deposit = ev::deposit_weighted_vlasov_pic_parallel(
        vp_markers, grid.mesh, step_options.shape, options.threads);
    ev::VlasovPoissonField vp_field =
        ev::solve_weighted_pic_poisson_field(
            vp_deposit, grid.mesh, time, vp_parameters);
    RadialDiagnostics ev_radial = radial_diagnostics(ev_markers);
    RadialDiagnostics vp_radial = radial_diagnostics(vp_markers);
    if (!options.resume) {
        summary.initial_compactness = ev_field.maximum_compactness;
        summary.maximum_compactness = ev_field.maximum_compactness;
        summary.maximum_compactness_time = time;
        summary.minimum_metric_gap = ev_field.metric.min_gap;
        initial_adm_mass = ev_field.adm_mass;
        summary.initial_virial_ratio = ev_radial.virial_ratio;
        summary.initial_r50 = ev_radial.r50;
        minimum_r50 = ev_radial.r50;
    } else {
        if (ev_field.maximum_compactness > summary.maximum_compactness) {
            summary.maximum_compactness = ev_field.maximum_compactness;
            summary.maximum_compactness_time = time;
        }
        summary.minimum_metric_gap = std::min(
            summary.minimum_metric_gap, ev_field.metric.min_gap);
        summary.maximum_adm_mass_relative_drift = std::max(
            summary.maximum_adm_mass_relative_drift,
            std::abs(ev_field.adm_mass - initial_adm_mass)
                / initial_adm_mass);
        minimum_r50 = std::min(minimum_r50, ev_radial.r50);
    }
    if (!options.resume) {
        write_low_restart_pair(
            options.output_dir, ev_markers, vp_markers, time, step);
    }
    const bool checkpoint_already_in_output =
        options.resume
        && last_csv_step(history_path) == step
        && last_csv_step(profiles_path) == step;
    if (!checkpoint_already_in_output) {
        const double density_l1 = shell_mass_l1_difference(
            ev_markers, vp_markers, grid.mesh, step_options.shape);
        const auto rms = options.resume
            ? particle_rms_difference(ev_markers, vp_markers)
            : std::pair<double, double>{0.0, 0.0};
        write_low_row(
            history,
            step,
            time,
            0.0,
            ev_field,
            vp_field,
            ev_radial,
            vp_radial,
            density_l1,
            rms.first,
            rms.second);
        write_low_profile(
            profiles, step, time, grid, ev_field, vp_deposit, vp_field);
        ev::flush_output_checked(history, "low compactness history");
        ev::flush_output_checked(profiles, "low compactness profiles");
    }

    auto save_low_accepted_state = [&](double dt, bool append_diagnostics) {
        write_low_restart_pair(
            options.output_dir, ev_markers, vp_markers, time, step);
        if (!append_diagnostics) return;
        ev_radial = radial_diagnostics(ev_markers);
        vp_radial = radial_diagnostics(vp_markers);
        minimum_r50 = std::min(minimum_r50, ev_radial.r50);
        const double density_l1 = shell_mass_l1_difference(
            ev_markers, vp_markers, grid.mesh, step_options.shape);
        const auto rms = particle_rms_difference(ev_markers, vp_markers);
        write_low_row(history, step, time, dt, ev_field, vp_field,
            ev_radial, vp_radial, density_l1, rms.first, rms.second);
        write_low_profile(profiles, step, time, grid, ev_field, vp_deposit, vp_field);
        ev::flush_output_checked(history, "low compactness history");
        ev::flush_output_checked(profiles, "low compactness profiles");
        std::cout << "  low step=" << step
                  << " t=" << std::setprecision(5) << time
                  << " Cmax=" << ev_field.maximum_compactness
                  << " Q=" << ev_radial.virial_ratio
                  << " r50=" << ev_radial.r50
                  << " EV/VP_L1=" << density_l1 << '\n';
    };
    double last_accepted_dt = 0.0;
    while (time < options.low_t_end
           && ev_field.metric.min_gap > ev_parameters.gap_stop) {
        double dt = ev::choose_weighted_einstein_vlasov_pic_timestep(
            ev_markers,
            grid,
            ev_field.metric,
            options.low_cfl,
            options.low_dt_max);
        dt = std::min(dt, options.low_t_end - time);
        ev::WeightedEinsteinVlasovPICStepResult ev_step;
        try {
            ev_step = ev::advance_weighted_einstein_vlasov_pic_self_consistent(
                ev_markers, dt, grid, step_options);
        } catch (const ev::WeightedEinsteinVlasovPICVacuumOuterBoundary& error) {
            summary.schwarzschild_evolution.outer_domain_limited = true;
            summary.schwarzschild_evolution.rejected_trial_dt = dt;
            summary.schwarzschild_evolution.termination_detail = error.what();
            save_low_accepted_state(last_accepted_dt,
                last_csv_step(history_path) < step || last_csv_step(profiles_path) < step);
            break;
        }
        // The relativistic step may shorten a requested interval near its
        // metric limit.  Advance VP only after that interval is accepted,
        // using the returned duration for both ends of its KDK update.
        dt = ev_step.accepted_dt;
        ev_field = ev_step.endpoint_field;
        const double endpoint_time = time + dt;
        if (ev_field.maximum_compactness > summary.maximum_compactness) {
            summary.maximum_compactness = ev_field.maximum_compactness;
            summary.maximum_compactness_time = endpoint_time;
        }
        summary.minimum_metric_gap = std::min(
            summary.minimum_metric_gap,
            ev_field.metric.min_gap);
        summary.maximum_adm_mass_relative_drift = std::max(
            summary.maximum_adm_mass_relative_drift,
            std::abs(ev_field.adm_mass - initial_adm_mass)
                / initial_adm_mass);

        // Newtonian KDK uses the old force for the opening kick, deposits
        // the drifted positions to solve Poisson, then uses that endpoint
        // force for the closing kick.
        ev::begin_weighted_pic_self_consistent_kdk_parallel(
            vp_markers,
            time,
            dt,
            grid.radial_coordinate,
            grid.radius,
            vp_field,
            vp_parameters,
            options.threads);
        vp_deposit = ev::deposit_weighted_vlasov_pic_parallel(
            vp_markers, grid.mesh, step_options.shape, options.threads);
        vp_field = ev::solve_weighted_pic_poisson_field(
            vp_deposit, grid.mesh, time + dt, vp_parameters);
        ev::finish_weighted_pic_self_consistent_kdk_parallel(
            vp_markers,
            time + dt,
            dt,
            grid.radial_coordinate,
            grid.radius,
            vp_field,
            vp_parameters,
            options.threads);

        time += dt;
        ++step;
        last_accepted_dt = dt;
        if (step % options.low_output_steps == 0
            || ev_field.metric.min_gap <= ev_parameters.gap_stop
            || time >= options.low_t_end) {
            save_low_accepted_state(dt, true);
        }
    }

    ev_radial = radial_diagnostics(ev_markers);
    vp_radial = radial_diagnostics(vp_markers);
    summary.final_compactness = ev_field.maximum_compactness;
    summary.final_virial_ratio = ev_radial.virial_ratio;
    summary.minimum_r50 = minimum_r50;
    summary.final_r50 = ev_radial.r50;
    summary.density_l1_difference = shell_mass_l1_difference(
        ev_markers, vp_markers, grid.mesh, step_options.shape);
    const auto rms = particle_rms_difference(ev_markers, vp_markers);
    summary.particle_rms_radius_difference = rms.first;
    summary.particle_rms_momentum_difference = rms.second;
    summary.ev_rest_mass_relative_error = std::abs(
        ev::weighted_pic_total_marker_mass(ev_markers)
        - initial_rest_mass) / initial_rest_mass;
    summary.vp_rest_mass_relative_error = std::abs(
        ev::weighted_pic_total_marker_mass(vp_markers)
        - initial_vp_mass) / initial_vp_mass;
    summary.schwarzschild_evolution.final_time = time;
    summary.virialized_halo =
        !summary.schwarzschild_evolution.outer_domain_limited
        && summary.minimum_metric_gap > ev_parameters.gap_stop
        && summary.maximum_compactness < 1.0
        && summary.minimum_r50 < 0.9 * summary.initial_r50
        && summary.final_r50 < 1.5 * summary.initial_r50
        && summary.final_virial_ratio > 0.55
        && summary.final_virial_ratio < 1.45;
    summary.agrees_with_vlasov_poisson =
        !summary.schwarzschild_evolution.outer_domain_limited
        && summary.final_compactness < 0.02
        && summary.density_l1_difference < 0.15
        && summary.particle_rms_radius_difference < 0.15
        && summary.particle_rms_momentum_difference < 0.03;
    return summary;
}

// ---- Dynamic null-slice extraction from a Schwarzschild donor ------------
struct AcceptedDonorStep {
    double reverse_time_before = 0.0;
    double requested_dt = 0.0;
    double accepted_dt = 0.0;
    int gap_bisections = 0;
    std::uint64_t marker_fingerprint = 0;
    std::uint64_t field_fingerprint = 0;
};

struct NullExtractionResult {
    std::string manifest_sha256;
    double boundary_radius = 0.0;
    double evolution_outer_radius = 0.0;
    double ef_v_offset = 0.0;
    double null_slice_conserved_adm_mass = 0.0;
    std::uint64_t ef_realized_fingerprint = 0;
    bool analytic_vacuum_extension = false;
};

std::uint64_t schwarzschild_field_fingerprint(
    const ev::WeightedEinsteinVlasovPICField& field) {
    FNV64Fingerprint hash;
    auto mix_vector = [&](const std::vector<double>& value) {
        hash.mix_u64(static_cast<std::uint64_t>(value.size()));
        for (double entry : value) {
            hash.mix_double(entry);
        }
    };
    hash.mix_double(field.rest_mass);
    hash.mix_double(field.adm_mass);
    hash.mix_double(field.maximum_compactness);
    mix_vector(field.metric.lambda);
    mix_vector(field.metric.mu);
    mix_vector(field.metric.mu_r);
    mix_vector(field.metric.m);
    mix_vector(field.metric.face_radius);
    mix_vector(field.metric.face_lambda);
    mix_vector(field.metric.face_mu);
    mix_vector(field.metric.face_mu_r);
    mix_vector(field.moments.rho);
    mix_vector(field.moments.p);
    mix_vector(field.moments.pt);
    mix_vector(field.moments.j);
    return hash.value();
}

void write_dense_donor_history(
    const std::filesystem::path& path,
    const ev::coordinate_map::SchwarzschildHistory& history) {
    const auto nonce = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const std::filesystem::path temporary =
        path.string() + ".tmp." + std::to_string(nonce);
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output || history.snapshot.empty()) {
            throw std::runtime_error("Could not create dense donor history.");
        }
        const char magic[16] = {
            'V','E','N','U','S','S','C','H','I','S','T','1',0,0,0,0,
        };
        output.write(magic, sizeof(magic));
        // Version 2 adds the exact face reconstruction consumed by
        // sample_metric_on_grid.  Centre-only version-1 histories force an
        // independent comparator to solve the eikonal equation on a
        // different discrete spacetime and produce a non-convergent
        // Jacobian-composition floor.
        write_physical_ic_scalar<std::uint32_t>(output, 2);
        write_physical_ic_scalar<std::uint64_t>(
            output, static_cast<std::uint64_t>(history.snapshot.size()));
        write_physical_ic_scalar<std::uint64_t>(
            output,
            static_cast<std::uint64_t>(
                history.snapshot.front().field.metric.lambda.size()));
        for (const auto& snapshot : history.snapshot) {
            const std::size_t cells =
                snapshot.field.metric.lambda.size();
            if (snapshot.field.metric.mu.size() != cells
                || snapshot.field.metric.mu_r.size() != cells
                || snapshot.field.metric.face_radius.size() != cells + 1
                || snapshot.field.metric.face_lambda.size() != cells + 1
                || snapshot.field.metric.face_mu.size() != cells + 1
                || snapshot.field.metric.face_mu_r.size() != cells + 1) {
                throw std::runtime_error(
                    "Dense donor history lacks production metric faces.");
            }
            write_physical_ic_scalar(output, snapshot.reverse_time);
            write_physical_ic_scalar(output, snapshot.field.rest_mass);
            write_physical_ic_scalar(output, snapshot.field.adm_mass);
            write_physical_ic_scalar(
                output, snapshot.field.maximum_compactness);
            write_physical_ic_scalar(output, snapshot.field.metric.min_gap);
            const std::vector<double>* array[] = {
                &snapshot.field.metric.lambda,
                &snapshot.field.metric.mu,
                &snapshot.field.metric.m,
                &snapshot.field.moments.rho,
                &snapshot.field.moments.p,
                &snapshot.field.moments.pt,
                &snapshot.field.moments.j,
            };
            for (const auto* value : array) {
                for (double entry : *value) {
                    write_physical_ic_scalar(output, entry);
                }
            }
            const std::vector<double>* reconstruction[] = {
                &snapshot.field.metric.face_radius,
                &snapshot.field.metric.face_lambda,
                &snapshot.field.metric.face_mu,
                &snapshot.field.metric.face_mu_r,
                &snapshot.field.metric.mu_r,
            };
            for (const auto* value : reconstruction) {
                for (double entry : *value) {
                    write_physical_ic_scalar(output, entry);
                }
            }
        }
        output.flush();
        if (!output) {
            throw std::runtime_error("Could not flush dense donor history.");
        }
        output.close();
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

std::string json_escape(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (char character : value) {
        if (character == '"' || character == '\\') {
            result.push_back('\\');
        }
        result.push_back(character);
    }
    return result;
}

// Build EF data on one ingoing null hypersurface from a physical SC state.
// A null slice intersects different worldlines at different SC times.  First
// generate the required SC history, solve the ingoing eikonal surface through
// it, then locate each worldline crossing and transform its momentum there.
NullExtractionResult extract_ef_null_initial_data(
    const RunOptions& options,
    const ev::Params& parameters,
    const ev::WeightedPICFieldGrid& grid,
    const std::vector<ev::WeightedPICMarker>& physical_markers,
    std::uint64_t base_ic_fingerprint) {
    namespace map = ev::coordinate_map;
    if (options.export_ef_null_initial_data_path.empty()
        || physical_markers.empty()
        || parameters.Nr != static_cast<int>(grid.radius.size())) {
        throw std::runtime_error("Invalid null extraction request.");
    }
    const double evolution_outer_radius = grid.mesh.edge.back();
    const double requested_map_worldtube =
        options.ef_map_worldtube_radius > 0.0
            ? options.ef_map_worldtube_radius : evolution_outer_radius;
    const std::size_t map_cell_count = null_map_worldtube_cell_count(
        grid, requested_map_worldtube);
    const bool analytic_vacuum_extension =
        map_cell_count < grid.radius.size();
    const double map_local_dr = grid.mesh.edge[map_cell_count]
        - grid.mesh.edge[map_cell_count - 1];
    const double required_worldtube_clearance = 4.0 * map_local_dr;
    std::unordered_set<std::uint64_t> identifiers;
    identifiers.reserve(physical_markers.size());
    for (const auto& marker : physical_markers) {
        if (marker.state != ev::WeightedPICMarkerState::active
            || !identifiers.insert(marker.id).second) {
            throw std::runtime_error(
                "Null extraction requires unique active marker IDs.");
        }
    }
    const std::uint64_t sc_realized_fingerprint =
        physical_ic_fingerprint(physical_markers);
    // Reversing w constructs the time-reversed donor.  Positive reverse
    // time follows the physical solution into its past, supplying the part
    // of the ingoing null surface that precedes the original spacelike slice.
    std::vector<ev::WeightedPICMarker> reverse_initial = physical_markers;
    for (auto& marker : reverse_initial) {
        marker.w = -marker.w;
    }
    std::vector<ev::WeightedPICMarker> donor_markers = reverse_initial;
    double maximum_donor_marker_radius = 0.0;
    for (const auto& marker : donor_markers) {
        maximum_donor_marker_radius = std::max(
            maximum_donor_marker_radius, marker.r);
    }
    ev::WeightedEinsteinVlasovPICOptions evolution;
    evolution.shape = ev::WeightedPICParticleShape::tsc;
    evolution.threads = options.threads;
    evolution.characteristic_substeps = 2;
    evolution.require_vacuum_outer_boundary = true;

    map::SchwarzschildHistory history;
    ev::WeightedEinsteinVlasovPICField field =
        ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
            donor_markers, grid, evolution.shape, evolution.threads);
    if (!(field.metric.min_gap > 0.0)) {
        throw std::runtime_error(
            "The Schwarzschild donor starts outside its coordinate patch.");
    }
    const double conserved_adm_mass = field.adm_mass;
    const FixedFamilyBoundDiagnostics initial_bound =
        fixed_family_bound_diagnostics(
            physical_markers, grid, field);
    history.snapshot.push_back({0.0, field});
    std::vector<AcceptedDonorStep> accepted_steps;
    double reverse_time = 0.0;
    double donor_duration = options.ef_null_donor_duration > 0.0
        ? options.ef_null_donor_duration : options.high_r_max;
    const double maximum_automatic_duration = 3.0 * options.high_r_max;
    map::EikonalSurface surface;
    bool surface_solved = false;
    for (;;) {
        while (reverse_time < donor_duration) {
            double step_start_maximum_radius = 0.0;
            for (const auto& marker : donor_markers) {
                step_start_maximum_radius = std::max(
                    step_start_maximum_radius, marker.r);
            }
            double requested_dt =
                ev::choose_weighted_einstein_vlasov_pic_timestep(
                    donor_markers,
                    grid,
                    field.metric,
                    0.22,
                    options.high_dt_max);
            requested_dt = std::min(
                requested_dt, donor_duration - reverse_time);
            const auto result =
                ev::advance_weighted_einstein_vlasov_pic_self_consistent(
                    donor_markers, requested_dt, grid, evolution);
            if (!(result.accepted_dt > 0.0)) {
                throw std::runtime_error(
                    "The reverse Schwarzschild donor accepted zero dt.");
            }
            // For regular polar-areal matter, |dr/dt|=
            // exp(mu-lambda)|w|/epsilon <= 1.  Include this per-step causal
            // upper bound, not merely saved endpoint radii, in the proof
            // that the entire reverse-donor support stays inside R_B.
            maximum_donor_marker_radius = std::max(
                maximum_donor_marker_radius,
                step_start_maximum_radius + result.accepted_dt);
            for (const auto& marker : donor_markers) {
                if (marker.state != ev::WeightedPICMarkerState::active) {
                    throw std::runtime_error(
                        "A marker left the finite Schwarzschild domain "
                        "before intersecting the null slice; enlarge r_max.");
                }
                maximum_donor_marker_radius = std::max(
                    maximum_donor_marker_radius, marker.r);
            }
            accepted_steps.push_back({
                reverse_time,
                requested_dt,
                result.accepted_dt,
                result.gap_bisections,
                physical_ic_fingerprint(donor_markers),
                schwarzschild_field_fingerprint(result.endpoint_field),
            });
            reverse_time += result.accepted_dt;
            field = result.endpoint_field;
            history.snapshot.push_back({reverse_time, field});
        }
        // The eikonal equation makes the hypersurface null in the evolving
        // donor metric.  Its outer vacuum normalization fixes the advanced
        // time.  Extend the donor history if this surface needs earlier data.
        try {
            surface = map::solve_ingress_eikonal_surface(
                history, grid, reverse_time, 8,
                requested_map_worldtube);
            surface_solved = true;
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            const bool coverage_failure =
                message.find("donor history") != std::string::npos
                || message.find("extends before") != std::string::npos;
            if (options.ef_null_donor_duration > 0.0 || !coverage_failure
                || donor_duration + 0.25 * options.high_r_max
                    > maximum_automatic_duration) {
                throw;
            }
            donor_duration += 0.25 * options.high_r_max;
        }
        if (surface_solved) {
            break;
        }
    }

    // The worldtube fixing A=1 must be vacuum.  In a vacuum Schwarzschild
    // exterior lambda+mu=0, so the mapped b equals the EF normalization
    // b(r_outer)=1.  Check both the stress and normalization at every time.
    double maximum_outer_source = 0.0;
    double maximum_source = 0.0;
    double maximum_outer_b_error = 0.0;
    const std::size_t cells = grid.radius.size();
    const std::size_t outer_begin = map_cell_count > 4
        ? map_cell_count - 4 : 0;
    for (const auto& snapshot : history.snapshot) {
        const auto& moments = snapshot.field.moments;
        for (std::size_t i = 0; i < cells; ++i) {
            const double source = std::max({
                std::abs(moments.rho[i]), std::abs(moments.p[i]),
                std::abs(moments.pt[i]), std::abs(moments.j[i]),
            });
            maximum_source = std::max(maximum_source, source);
            if (i >= outer_begin && i < map_cell_count) {
                maximum_outer_source = std::max(
                    maximum_outer_source, source);
            }
        }
        const ev::MetricSample outer =
            ev::weighted_einstein_vlasov_pic_detail::sample_metric_on_grid(
                surface.outer_radius, grid, snapshot.field.metric);
        maximum_outer_b_error = std::max(
            maximum_outer_b_error,
            std::abs(std::exp(outer.lambda + outer.mu) - 1.0));
    }
    const double source_tolerance = 1.0e-13
        * std::max(maximum_source, 1.0e-300);
    // The radial metric constraint is second-order on this cell-centered
    // mesh.  Authenticate an explicit O(dr^2) normalization allowance at
    // R_B, with a tight roundoff/converged floor, rather than pretending a
    // coarse discrete vacuum has continuum-level b=1 accuracy.
    const double outer_b_tolerance = std::max(
        2.0e-10, 1.0e-5 * map_local_dr * map_local_dr);
    const double minimum_worldtube_matter_clearance =
        surface.outer_radius - maximum_donor_marker_radius;
    if (maximum_outer_source > source_tolerance
        || maximum_outer_b_error > outer_b_tolerance
        || minimum_worldtube_matter_clearance
            < required_worldtube_clearance) {
        std::ostringstream message;
        message << std::setprecision(17)
            << "The eikonal outer worldtube is not verified vacuum with "
               "b=1 normalization and four-cell donor-matter clearance: "
            << "maximum_source=" << maximum_outer_source
            << " (tolerance=" << source_tolerance << "), maximum_|b-1|="
            << maximum_outer_b_error << " (tolerance="
            << outer_b_tolerance << ')'
            << ", matter_clearance="
            << minimum_worldtube_matter_clearance
            << " (required=" << required_worldtube_clearance << "). "
               "Increase --ef-map-worldtube-radius or reduce the explicit "
               "donor duration.";
        throw std::runtime_error(message.str());
    }

    // Deterministic replay: use the exact accepted dt ledger, require bitwise
    // state/field fingerprints, and locate every worldline crossing once.
    std::vector<ev::WeightedPICMarker> replay = reverse_initial;
    std::vector<bool> crossed(replay.size(), false);
    std::vector<map::MarkerCrossing> crossings;
    crossings.reserve(replay.size());
    double replay_time = 0.0;
    double maximum_replay_time_error = 0.0;
    for (std::size_t step = 0; step < accepted_steps.size(); ++step) {
        const auto before = replay;
        const AcceptedDonorStep& ledger = accepted_steps[step];
        const auto result =
            ev::advance_weighted_einstein_vlasov_pic_self_consistent(
                replay, ledger.accepted_dt, grid, evolution);
        maximum_replay_time_error = std::max(
            maximum_replay_time_error,
            std::abs(result.accepted_dt - ledger.accepted_dt));
        if (result.gap_bisections != 0
            || result.accepted_dt != ledger.accepted_dt
            || physical_ic_fingerprint(replay) != ledger.marker_fingerprint
            || schwarzschild_field_fingerprint(result.endpoint_field)
                != ledger.field_fingerprint) {
            throw std::runtime_error(
                "The Schwarzschild donor replay is not deterministic; "
                "refusing an unauditable coordinate map.");
        }
        const double next_time = replay_time + result.accepted_dt;
        // A change of sign of the crossing function brackets the event
        // where this worldline meets the null surface.  Interpolate within
        // that accepted step and mark the ID so it is used exactly once.
        for (std::size_t q = 0; q < replay.size(); ++q) {
            const double g0 = map::crossing_function(
                replay_time, before[q].r, surface);
            const double g1 = map::crossing_function(
                next_time, replay[q].r, surface);
            const double tolerance = 512.0
                * std::numeric_limits<double>::epsilon()
                * std::max({1.0, std::abs(replay_time),
                    std::abs(next_time)});
            if (!crossed[q] && g0 <= tolerance && g1 >= -tolerance) {
                crossings.push_back(map::interpolate_marker_crossing(
                    before[q], replay[q], replay_time, next_time,
                    surface, history, grid, step + 1));
                crossed[q] = true;
            } else if (!crossed[q] && g0 > tolerance) {
                throw std::runtime_error(
                    "A marker crossing precedes the replay history.");
            } else if (crossed[q] && g1 < -tolerance) {
                throw std::runtime_error(
                    "A marker crossed the same ingoing null slice more "
                    "than once; time resolution/orientation is invalid.");
            }
        }
        replay_time = next_time;
    }
    if (crossings.size() != physical_markers.size()
        || !std::all_of(crossed.begin(), crossed.end(), [](bool value) {
            return value;
        })) {
        throw std::runtime_error(
            "Not every marker crossed the solved ingoing null slice once; "
            "increase donor duration or enlarge the radial domain.");
    }
    std::sort(
        crossings.begin(), crossings.end(),
        [](const map::MarkerCrossing& first,
           const map::MarkerCrossing& second) {
            return first.ef_marker.id < second.ef_marker.id;
        });
    for (std::size_t q = 1; q < crossings.size(); ++q) {
        if (crossings[q - 1].ef_marker.id == crossings[q].ef_marker.id) {
            throw std::runtime_error("Duplicate marker ID in crossing audit.");
        }
    }
    double maximum_crossing_residual = 0.0;
    double maximum_crossing_radius = 0.0;
    std::vector<ev::eddington_finkelstein::Marker> ef_markers;
    ef_markers.reserve(crossings.size());
    for (const auto& crossing : crossings) {
        maximum_crossing_residual = std::max(
            maximum_crossing_residual, std::abs(crossing.residual));
        maximum_crossing_radius = std::max(
            maximum_crossing_radius, crossing.ef_marker.r);
        ef_markers.push_back(crossing.ef_marker);
    }
    if (maximum_crossing_residual
        > 1.0e-10 * std::max(1.0, reverse_time)) {
        throw std::runtime_error(
            "Same-event marker crossing residual exceeds the extraction "
            "tolerance; reduce --high-dt-max.");
    }
    const double minimum_crossing_radius_clearance =
        surface.outer_radius - maximum_crossing_radius;
    if (minimum_crossing_radius_clearance
        < required_worldtube_clearance) {
        throw std::runtime_error(
            "Mapped marker crossings lack the required four-cell vacuum "
            "clearance from the authenticated worldtube; increase "
            "--ef-map-worldtube-radius.");
    }
    // Transform stress tensors and metric at points on the solved surface.
    // Beyond the verified vacuum worldtube the continuation is analytic
    // Schwarzschild vacuum with the ADM mass of this same null slice.
    std::vector<map::MappedSlicePoint> mapped_slice =
        map::map_schwarzschild_slice_tensors(history, surface, grid);
    if (mapped_slice.size() != map_cell_count) {
        throw std::runtime_error(
            "The numerical mapped slice is not the authenticated grid "
            "prefix ending at the worldtube.");
    }
    const std::size_t analytic_extension_start_index = mapped_slice.size();
    const double null_slice_adm_mass =
        history.snapshot.back().field.adm_mass;
    map::SchwarzschildVacuumExtensionAudit vacuum_audit;
    if (analytic_vacuum_extension) {
        mapped_slice = map::extend_mapped_slice_with_schwarzschild_vacuum(
            mapped_slice, grid.radius, surface.outer_radius,
            surface.outer_reverse_time, null_slice_adm_mass);
        vacuum_audit = map::audit_schwarzschild_vacuum_extension(
            mapped_slice, analytic_extension_start_index,
            surface.outer_radius, surface.outer_reverse_time,
            null_slice_adm_mass);
    }

    EFNullInitialData null_data;
    null_data.markers = ef_markers;
    null_data.mapped_slice = mapped_slice;
    // Anchor the EF radial constraint to the ADM mass on this same null
    // slice.  The initial-to-slice donor drift remains an explicit audit
    // below; using the t=0 value here would itself create a boundary mismatch.
    null_data.conserved_adm_mass = null_slice_adm_mass;
    null_data.donor_duration = reverse_time;
    null_data.outer_radius = evolution_outer_radius;
    null_data.map_worldtube_radius = surface.outer_radius;
    null_data.outer_advanced_time = surface.outer_advanced_time;
    null_data.ef_v_offset = reverse_time;
    null_data.maximum_eikonal_residual = surface.maximum_ode_residual;
    null_data.maximum_crossing_residual = maximum_crossing_residual;
    null_data.minimum_crossing_radius_clearance =
        minimum_crossing_radius_clearance;
    null_data.minimum_worldtube_matter_clearance =
        minimum_worldtube_matter_clearance;
    null_data.maximum_worldtube_source = maximum_outer_source;
    null_data.maximum_worldtube_b_normalization_error =
        maximum_outer_b_error;
    null_data.worldtube_b_normalization_tolerance = outer_b_tolerance;
    null_data.maximum_analytic_vacuum_field_residual =
        vacuum_audit.maximum_field_residual;
    null_data.maximum_analytic_vacuum_source =
        vacuum_audit.maximum_source;
    null_data.maximum_analytic_inverse_map_residual =
        vacuum_audit.maximum_inverse_map_residual;
    null_data.minimum_analytic_vacuum_a = vacuum_audit.minimum_a;
    null_data.base_ic_fingerprint = base_ic_fingerprint;
    null_data.schwarzschild_realized_fingerprint = sc_realized_fingerprint;
    null_data.ef_realized_fingerprint = ef_marker_fingerprint(ef_markers);
    null_data.grid_fingerprint = coordinate_map_grid_fingerprint(grid);
    null_data.map_grid_fingerprint =
        coordinate_map_grid_prefix_fingerprint(grid, map_cell_count);
    null_data.sampling_realization = options.high_sampling_realization;
    null_data.analytic_extension_start_index =
        analytic_extension_start_index;
    null_data.analytic_vacuum_extension = analytic_vacuum_extension;

    const std::filesystem::path base =
        options.export_ef_null_initial_data_path;
    const std::filesystem::path crossings_path =
        null_artifact_path(base, ".crossings.csv");
    const std::filesystem::path slice_path =
        null_artifact_path(base, ".slice.csv");
    const std::filesystem::path steps_path =
        null_artifact_path(base, ".accepted_steps.csv");
    const std::filesystem::path history_path =
        null_artifact_path(base, ".donor_history.bin");
    const std::filesystem::path manifest_path =
        null_artifact_path(base, ".manifest.json");
    // Save the EF particles with the mapped slice, accepted donor intervals
    // and crossing events.  Their fingerprints and file hashes connect each
    // transformed momentum to the worldline and metric used to compute it.
    write_null_initial_data(base, null_data);
    write_dense_donor_history(history_path, history);

    {
        std::ofstream output(crossings_path);
        if (!output) {
            throw std::runtime_error("Could not create marker crossing audit.");
        }
        output << std::setprecision(17)
            << "id,source_id,rest_weight,eta,reverse_time,schwarzschild_time,"
               "r,w_reverse,w_schwarzschild,p_r_ef,lambda,mu,"
               "surface_reverse_time,residual,replay_step,step_fraction,"
               "source_marker_fingerprint,ef_marker_fingerprint\n";
        for (const auto& crossing : crossings) {
            ev::WeightedPICMarker source;
            source.mass = crossing.ef_marker.rest_weight;
            source.r = crossing.ef_marker.r;
            source.w = crossing.w_schwarzschild;
            source.eta = crossing.ef_marker.eta;
            source.id = crossing.ef_marker.id;
            source.source_cell = crossing.ef_marker.source_id;
            std::vector<ev::WeightedPICMarker> one_source{source};
            std::vector<ev::eddington_finkelstein::Marker> one_ef{
                crossing.ef_marker};
            output << crossing.ef_marker.id << ','
                   << crossing.ef_marker.source_id << ','
                   << crossing.ef_marker.rest_weight << ','
                   << crossing.ef_marker.eta << ','
                   << crossing.reverse_time << ','
                   << crossing.schwarzschild_time << ','
                   << crossing.ef_marker.r << ','
                   << crossing.w_reverse << ','
                   << crossing.w_schwarzschild << ','
                   << crossing.ef_marker.p_r << ','
                   << crossing.lambda << ',' << crossing.mu << ','
                   << crossing.surface_reverse_time << ','
                   << crossing.residual << ',' << crossing.replay_step << ','
                   << crossing.step_fraction << ",0x" << std::hex
                   << std::setw(16) << std::setfill('0')
                   << physical_ic_fingerprint(one_source) << ",0x"
                   << std::setw(16) << ef_marker_fingerprint(one_ef)
                   << std::dec << std::setfill(' ') << '\n';
        }
    }
    {
        std::ofstream output(slice_path);
        if (!output) {
            throw std::runtime_error("Could not create mapped slice audit.");
        }
        output << std::setprecision(17)
            << "r,reverse_time,schwarzschild_time,A,F,a,b,m,rho,p_radial,"
               "p_tangential_sum,j,T_vv,T_vr,T_rr,T_theta_theta,S\n";
        for (const auto& point : mapped_slice) {
            output << point.r << ',' << point.reverse_time << ','
                   << point.schwarzschild_time << ',' << point.A << ','
                   << point.F << ',' << point.a << ',' << point.b << ','
                   << point.m << ',' << point.rho << ',' << point.p << ','
                   << point.pt << ',' << point.j << ',' << point.t_vv << ','
                   << point.t_vr << ',' << point.t_rr << ','
                   << point.t_theta_theta << ',' << point.s << '\n';
        }
        // The audit reaches the exact EF evolution face.  For R_E>R_B this
        // is the same v=const hypersurface continued through static vacuum:
        // s=T+r_*(R_E)-r_*(R_B), A=F=b=1 and a=1-2M/R_E.
        const auto outer = map::make_schwarzschild_vacuum_extension_point(
            evolution_outer_radius, surface.outer_radius,
            surface.outer_reverse_time, null_slice_adm_mass);
        output << outer.r << ',' << outer.reverse_time << ','
               << outer.schwarzschild_time << ',' << outer.A << ','
               << outer.F << ',' << outer.a << ',' << outer.b << ','
               << outer.m << ",0,0,0,0,0,0,0,0,0\n";
    }
    {
        std::ofstream output(steps_path);
        if (!output) {
            throw std::runtime_error("Could not create accepted-step ledger.");
        }
        output << std::setprecision(17)
            << "step,reverse_time_before,requested_dt,accepted_dt,"
               "gap_bisections,marker_fingerprint,field_fingerprint\n";
        for (std::size_t i = 0; i < accepted_steps.size(); ++i) {
            const auto& step = accepted_steps[i];
            output << i + 1 << ',' << step.reverse_time_before << ','
                   << step.requested_dt << ',' << step.accepted_dt << ','
                   << step.gap_bisections << ",0x" << std::hex
                   << std::setw(16) << std::setfill('0')
                   << step.marker_fingerprint << ",0x" << std::setw(16)
                   << step.field_fingerprint << std::dec
                   << std::setfill(' ') << '\n';
        }
    }

    const std::string binary_sha = ev::integrity::sha256_file(base);
    const std::string crossings_sha =
        ev::integrity::sha256_file(crossings_path);
    const std::string slice_sha = ev::integrity::sha256_file(slice_path);
    const std::string steps_sha = ev::integrity::sha256_file(steps_path);
    const std::string history_sha = ev::integrity::sha256_file(history_path);
    {
        std::ofstream output(manifest_path);
        if (!output) {
            throw std::runtime_error("Could not create null-map manifest.");
        }
        output << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"venus_v3_sc_to_ef_null_map\",\n"
            << "  \"schema_version\": 2,\n"
            << "  \"complete\": true,\n"
            << "  \"all_markers_crossed_once\": true,\n"
            << "  \"outer_worldtube_verified_vacuum\": true,\n"
            << "  \"mapping_scope\": \"initial_null_hypersurface\",\n"
            << "  \"donor_direction\": \"East_time_reversed_outgoing_s=-tau\",\n"
            << "  \"physical_schwarzschild_run\": \"incoming markers are unmodified; reverse donor is an independent copy\",\n"
            << "  \"time_reversal_momentum\": \"w_reverse=-w_schwarzschild\",\n"
            << "  \"eikonal_equation\": \"tau_sigma_r=-exp(lambda-mu)\",\n"
            << "  \"reverse_eikonal_equation\": \"s_sigma_r=exp(lambda-mu)\",\n"
            << "  \"integrating_factor_equation\": \"d_ln_A_dr=-partial_s_exp(lambda-mu)=partial_tau_exp(lambda-mu)\",\n"
            << "  \"eikonal_integration_method\": \"per_cell_RK4_step_doubling_n_vs_2n_accept_2n\",\n"
            << "  \"eikonal_richardson_error_normalization\": \"abs(Y_2n-Y_n)/(15*max(1,abs(Y_in),abs(Y_2n)));Y={s_sigma,ln_A}\",\n"
            << "  \"donor_history_format_version\": 2,\n"
            << "  \"saved_metric_interpolation\": \"SC center-face lambda plus monotone-Hermite mu; EF center-face a,b\",\n"
            << "  \"marker_momentum_map\": \"P=exp(lambda)*(epsilon+w_schwarzschild)>0\",\n"
            << "  \"outer_normalization\": \"at R_B: v=tau, A=v_tau=1, F=tau_v=1, b=1\",\n"
            << "  \"ef_time_coordinate\": \"v_bar=v+ef_v_offset\",\n"
            << "  \"outer_b_normalization\": 1,\n"
            << "  \"outer_boundary_radius\": " << surface.outer_radius
            << ",\n"
            << "  \"map_worldtube_radius\": " << surface.outer_radius
            << ",\n"
            << "  \"evolution_outer_radius\": "
            << evolution_outer_radius << ",\n"
            << "  \"analytic_vacuum_extension\": "
            << (analytic_vacuum_extension ? "true" : "false")
            << ",\n"
            << "  \"analytic_vacuum_extension_start_index\": "
            << analytic_extension_start_index << ",\n"
            << "  \"analytic_vacuum_model\": \"Schwarzschild: A=F=b=1, a=1-2M/r, m=M, T_ab=S=0\",\n"
            << "  \"analytic_inverse_map\": \"tau(r)|v=tau(R_B)-[r_*(r)-r_*(R_B)], d_tau/dr|v=-F/(a*b)=-1/a\",\n"
            << "  \"eikonal_outer_reverse_time\": " << reverse_time
            << ",\n"
            << "  \"eikonal_outer_schwarzschild_time\": "
            << -reverse_time << ",\n"
            << "  \"eikonal_outer_advanced_time\": "
            << surface.outer_advanced_time << ",\n"
            << "  \"ef_v_offset\": " << reverse_time << ",\n"
            << "  \"history_first_reverse_time\": 0,\n"
            << "  \"history_last_reverse_time\": " << reverse_time
            << ",\n"
            << "  \"history_snapshot_count\": "
            << history.snapshot.size() << ",\n"
            << "  \"donor_initial_adm_mass\": " << conserved_adm_mass
            << ",\n"
            << "  \"null_slice_adm_mass\": "
            << null_data.conserved_adm_mass << ",\n"
            << "  \"donor_adm_mass_relative_drift\": "
            << std::abs(null_data.conserved_adm_mass - conserved_adm_mass)
                / std::max(conserved_adm_mass, 1.0e-300) << ",\n"
            << "  \"accepted_step_count\": " << accepted_steps.size()
            << ",\n"
            << "  \"marker_count\": " << physical_markers.size() << ",\n"
            << "  \"crossing_count\": " << crossings.size() << ",\n"
            << "  \"unique_crossing_count\": " << crossings.size()
            << ",\n"
            << "  \"radial_cell_count\": " << grid.radius.size() << ",\n"
            << "  \"evolution_radial_cell_count\": "
            << grid.radius.size() << ",\n"
            << "  \"map_radial_cell_count\": "
            << map_cell_count << ",\n"
            << "  \"sampling_realization\": "
            << options.high_sampling_realization << ",\n"
            << "  \"target_initial_adm_mass\": "
            << options.high_target_initial_adm_mass << ",\n"
            << "  \"realized_initial_rest_weight\": "
            << ev::weighted_pic_total_marker_mass(physical_markers)
            << ",\n"
            << "  \"fixed_family_rest_weight\": "
            << initial_bound.rest_weight << ",\n"
            << "  \"fixed_family_binding_energy\": "
            << initial_bound.binding_energy << ",\n"
            << "  \"fixed_family_binding_fraction\": "
            << initial_bound.binding_fraction << ",\n"
            << "  \"fixed_family_positive_binding_energy\": "
            << (initial_bound.binding_fraction > 0.0
                ? "true" : "false") << ",\n"
            << "  \"initial_energy_at_infinity_proxy_definition\": "
               "\"E_proxy=exp(mu(t=0,r))*sqrt(1+w^2+2*eta/r^2);"
               " instantaneous frozen polar-areal metric; not conserved "
               "for dynamical data\",\n"
            << "  \"initial_energy_at_infinity_proxy_minimum\": "
            << initial_bound.energy_proxy_minimum << ",\n"
            << "  \"initial_energy_at_infinity_proxy_maximum\": "
            << initial_bound.energy_proxy_maximum << ",\n"
            << "  \"initial_energy_at_infinity_proxy_rest_weighted_mean\": "
            << initial_bound.energy_proxy_rest_weighted_mean << ",\n"
            << "  \"initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one\": "
            << initial_bound.energy_proxy_rest_weight_fraction_ge_one
            << ",\n"
            << "  \"base_ic_fingerprint\": \"0x" << std::hex
            << std::setw(16) << std::setfill('0') << base_ic_fingerprint
            << "\",\n"
            << "  \"schwarzschild_realized_fingerprint\": \"0x"
            << std::setw(16) << sc_realized_fingerprint << "\",\n"
            << "  \"ef_realized_fingerprint\": \"0x" << std::setw(16)
            << null_data.ef_realized_fingerprint << "\",\n"
            << "  \"grid_fingerprint\": \"0x" << std::setw(16)
            << null_data.grid_fingerprint << "\",\n"
            << "  \"map_grid_fingerprint\": \"0x" << std::setw(16)
            << null_data.map_grid_fingerprint << "\",\n"
            << "  \"null_map_fingerprint\": \"0x" << std::setw(16)
            << null_initial_data_fingerprint(null_data) << "\",\n"
            << std::dec << std::setfill(' ')
            << "  \"maximum_eikonal_reverse_time_richardson_error\": "
            << surface.maximum_reverse_time_richardson_error << ",\n"
            << "  \"maximum_eikonal_log_A_richardson_error\": "
            << surface.maximum_log_A_richardson_error << ",\n"
            << "  \"maximum_eikonal_ode_residual\": "
            << surface.maximum_ode_residual << ",\n"
            << "  \"maximum_marker_crossing_residual\": "
            << maximum_crossing_residual << ",\n"
            << "  \"maximum_donor_marker_radius\": "
            << maximum_donor_marker_radius << ",\n"
            << "  \"maximum_crossing_radius\": "
            << maximum_crossing_radius << ",\n"
            << "  \"minimum_worldtube_matter_clearance\": "
            << minimum_worldtube_matter_clearance << ",\n"
            << "  \"minimum_crossing_radius_clearance\": "
            << minimum_crossing_radius_clearance << ",\n"
            << "  \"required_worldtube_clearance\": "
            << required_worldtube_clearance << ",\n"
            << "  \"maximum_analytic_vacuum_field_residual\": "
            << vacuum_audit.maximum_field_residual << ",\n"
            << "  \"maximum_analytic_vacuum_source\": "
            << vacuum_audit.maximum_source << ",\n"
            << "  \"maximum_analytic_inverse_map_residual\": "
            << vacuum_audit.maximum_inverse_map_residual << ",\n"
            << "  \"minimum_analytic_vacuum_a\": "
            << vacuum_audit.minimum_a << ",\n"
            << "  \"maximum_replay_time_error\": "
            << maximum_replay_time_error << ",\n"
            << "  \"maximum_outer_source\": " << maximum_outer_source
            << ",\n"
            << "  \"maximum_outer_b_normalization_error\": "
            << maximum_outer_b_error << ",\n"
            << "  \"outer_b_normalization_tolerance\": "
            << outer_b_tolerance << ",\n"
            << "  \"artifacts\": {\n"
            << "    \"null_initial_data\": {\"path\": \""
            << json_escape(base.string()) << "\", \"sha256\": \""
            << binary_sha << "\"},\n"
            << "    \"crossings\": {\"path\": \""
            << json_escape(crossings_path.string())
            << "\", \"sha256\": \"" << crossings_sha << "\"},\n"
            << "    \"slice\": {\"path\": \""
            << json_escape(slice_path.string())
            << "\", \"sha256\": \"" << slice_sha << "\"},\n"
            << "    \"accepted_steps\": {\"path\": \""
            << json_escape(steps_path.string())
            << "\", \"sha256\": \"" << steps_sha << "\"},\n"
            << "    \"donor_history\": {\"path\": \""
            << json_escape(history_path.string())
            << "\", \"sha256\": \"" << history_sha << "\"}\n"
            << "  }\n"
            << "}\n";
    }
    verify_null_manifest_artifacts(base);
    verify_null_manifest_metadata(base, null_data);
    NullExtractionResult result;
    result.manifest_sha256 = ev::integrity::sha256_file(manifest_path);
    result.boundary_radius = surface.outer_radius;
    result.evolution_outer_radius = evolution_outer_radius;
    result.ef_v_offset = reverse_time;
    result.null_slice_conserved_adm_mass =
        null_data.conserved_adm_mass;
    result.ef_realized_fingerprint = null_data.ef_realized_fingerprint;
    result.analytic_vacuum_extension = analytic_vacuum_extension;
    std::cout << "  wrote solved-eikonal null data " << base
              << " markers=" << crossings.size()
              << " donor_s=" << reverse_time
              << " manifest_sha256=" << result.manifest_sha256 << '\n';
    return result;
}

// ---- Compact relativistic collapse experiment ----------------------------
// Follow the polar-areal branch from a fresh physical realization or an
// accepted checkpoint. Reconstruct the metric at each orbit stage, retain
// rest-weight accounting, and stop at the declared positive metric-gap limit.
HighSummary run_high_compactness_schwarzschild(const RunOptions& options) {
    std::cout << "Running Schwarzschild/polar-areal high-compactness "
              << options.high_initial_condition << " collapse with "
              << options.high_markers << " markers...\n";
    ev::Params parameters =
        base_parameters(
            options.high_radial_cells,
            options.high_r_min,
            options.high_r_max);
    parameters.gap_stop = options.high_gap_stop;
    parameters.vp_timestep_omitted_mass_fraction =
        options.timestep_omitted_mass_fraction;
    parameters.inner_boundary = options.high_inner_boundary;
    parameters.radial_grid = options.high_radial_grid;
    const ev::WeightedPICFieldGrid grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(
            parameters, parameters.Nr);
    const std::filesystem::path history_path =
        options.output_dir / "high_compactness_history.csv";
    const std::filesystem::path profiles_path =
        options.output_dir / "high_compactness_profiles.csv";
    const std::filesystem::path restart_path =
        options.output_dir / "high_einstein_vlasov.restart";
    std::vector<ev::WeightedPICMarker> markers;
    HighSummary summary;
    double time = 0.0;
    int step = 0;
    double initial_rest_mass = 0.0;
    double exported_null_slice_conserved_adm_mass = -1.0;
    if (options.resume) {
        const ev::WeightedPICRestartState restart =
            ev::read_weighted_pic_restart(restart_path);
        if (restart.step > std::numeric_limits<int>::max()
            || restart.markers.size() != options.high_markers) {
            throw std::runtime_error(
                "High restart state and configured marker count "
                "do not match.");
        }
        markers = restart.markers;
        time = restart.time;
        step = static_cast<int>(restart.step);
        const HighResumeHistory accumulated =
            read_high_resume_history(history_path);
        summary = accumulated.summary;
        const FixedFamilyBoundDiagnostics resumed_bound =
            read_fixed_family_bound_diagnostics(options.output_dir);
        validate_energy_cutoff_fixed_family_binding(options, resumed_bound);
        set_fixed_family_bound_diagnostics(summary, resumed_bound);
        initial_rest_mass = accumulated.initial_rest_mass;
        summary.initial_rest_weight = initial_rest_mass;
        if (last_csv_step(history_path) > step
            || last_csv_step(profiles_path) > step) {
            throw std::runtime_error(
                "High restart checkpoint is older than existing CSV output.");
        }
        std::cout << "  resumed high EV checkpoint at step=" << step
                  << " t=" << std::setprecision(17) << time << '\n';
    } else {
        markers = load_or_make_high_base_markers(
            options, parameters, grid, &summary);
        summary.base_ic_fingerprint = physical_ic_fingerprint(markers);
        apply_high_angular_momentum_controls(markers, options);
        const InitialADMNormalization normalization =
            normalize_high_initial_adm_mass(markers, options, grid);
        summary.initial_rest_weight_scale =
            normalization.rest_weight_scale;
        summary.fixed_family_adm_mass =
            normalization.achieved_adm_mass;
        summary.realized_ic_fingerprint = physical_ic_fingerprint(markers);
        initial_rest_mass =
            ev::weighted_pic_total_marker_mass(markers);
        summary.initial_rest_weight = initial_rest_mass;
    }
    if (options.resume && !options.physical_ic_path.empty()) {
        if (!std::filesystem::exists(options.physical_ic_path)) {
            throw std::runtime_error(
                "Configured physical IC is missing during high resume.");
        }
        auto base_markers = options.high_initial_condition
                == "energy_cutoff_shell"
            ? load_or_make_high_base_markers(
                options, parameters, grid, &summary)
            : read_physical_ic(options.physical_ic_path);
        summary.base_ic_fingerprint =
            physical_ic_fingerprint(base_markers);
        apply_high_angular_momentum_controls(base_markers, options);
        const InitialADMNormalization normalization =
            normalize_high_initial_adm_mass(base_markers, options, grid);
        validate_restart_initial_marker_invariants(markers, base_markers);
        summary.initial_rest_weight_scale =
            normalization.rest_weight_scale;
        summary.fixed_family_adm_mass =
            normalization.achieved_adm_mass;
        summary.realized_ic_fingerprint =
            physical_ic_fingerprint(base_markers);
        summary.initial_rest_weight =
            ev::weighted_pic_total_marker_mass(base_markers);
    }
    if (!options.export_ef_null_initial_data_path.empty()) {
        const NullExtractionResult extraction =
            extract_ef_null_initial_data(
                options,
                parameters,
                grid,
                markers,
                summary.base_ic_fingerprint);
        summary.eikonal_mapped_initial_data = true;
        summary.coordinate_map_manifest_sha256 =
            extraction.manifest_sha256;
        summary.eikonal_boundary_radius = extraction.boundary_radius;
        summary.eikonal_evolution_outer_radius =
            extraction.evolution_outer_radius;
        summary.eikonal_v_offset = extraction.ef_v_offset;
        summary.eikonal_analytic_vacuum_extension =
            extraction.analytic_vacuum_extension;
        exported_null_slice_conserved_adm_mass =
            extraction.null_slice_conserved_adm_mass;
        if (!(exported_null_slice_conserved_adm_mass > 0.0)
            || !std::isfinite(exported_null_slice_conserved_adm_mass)) {
            throw std::runtime_error(
                "Solved-eikonal export returned no finite positive "
                "null-slice conserved ADM mass.");
        }
    }
    ev::WeightedEinsteinVlasovPICOptions step_options;
    step_options.shape = ev::WeightedPICParticleShape::tsc;
    step_options.threads = options.threads;
    step_options.characteristic_substeps = 2;
    step_options.require_vacuum_outer_boundary = true;
    ev::check_weighted_einstein_vlasov_pic_vacuum_outer_boundary(markers);

    std::ofstream history = open_csv_output(
        history_path,
        options.resume,
        "step,time,dt,rest_mass,adm_mass,max_compactness,min_gap,"
        "min_lapse,max_log_redshift,r10,r50,r90,virial_ratio,"
        "anisotropic_jeans_relative_l1",
        "high_compactness_history.csv");
    std::ofstream profiles = open_csv_output(
        profiles_path,
        options.resume,
        "step,time,r,rho,compactness,m,a,lambda,mu,lambda_t,"
        "p_radial,p_tangential_sum,j,metric_sample_kind,metric_cell,"
        "face_left_r,face_right_r,face_left_lambda,face_right_lambda,"
        "center_mu_r,face_left_mu,face_right_mu,face_left_mu_r,"
        "face_right_mu_r",
        "high_compactness_profiles.csv");

    ev::WeightedEinsteinVlasovPICField field =
        ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
            markers, grid, step_options.shape, options.threads);
    RadialDiagnostics radial = radial_diagnostics(markers);
    AnisotropicJeansDiagnostics jeans =
        schwarzschild_jeans_diagnostics(grid, field);
    const AngularMomentumDiagnostics angular =
        angular_momentum_diagnostics(markers);
    const auto initial_lapse = lapse_diagnostics(field.metric);
    if (!options.resume) {
        const FixedFamilyBoundDiagnostics bound =
            fixed_family_bound_diagnostics(markers, grid, field);
        validate_energy_cutoff_fixed_family_binding(options, bound);
        set_fixed_family_bound_diagnostics(summary, bound);
        write_fixed_family_bound_diagnostics(options.output_dir, bound);
        summary.initial_min_gap = field.metric.min_gap;
        summary.initial_min_lapse = initial_lapse.first;
        summary.initial_compactness = field.maximum_compactness;
        summary.initial_adm_mass = field.adm_mass;
        summary.initial_deposited_adm_mass = field.adm_mass;
        if (options.high_target_initial_adm_mass > 0.0
            && std::abs(summary.initial_adm_mass
                    - options.high_target_initial_adm_mass)
                > 5.0e-11 * options.high_target_initial_adm_mass) {
            throw std::runtime_error(
                "Initial SC field no longer matches the fixed-ADM target.");
        }
        summary.initial_r50 = radial.r50;
        summary.minimum_r50 = radial.r50;
        summary.mass_weighted_mean_specific_angular_momentum = angular.mean;
        summary.mass_weighted_rms_specific_angular_momentum = angular.rms;
        summary.maximum_specific_angular_momentum = angular.maximum;
        write_wpic_restart(
            restart_path, markers, time, step);
    }
    summary.mass_weighted_mean_specific_angular_momentum = angular.mean;
    summary.mass_weighted_rms_specific_angular_momentum = angular.rms;
    summary.maximum_specific_angular_momentum = angular.maximum;
    const bool checkpoint_already_in_output =
        options.resume
        && last_csv_step(history_path) == step
        && last_csv_step(profiles_path) == step;
    if (!checkpoint_already_in_output) {
        history << step << ',' << time << ",0,"
                << field.rest_mass << ',' << field.adm_mass << ','
                << field.maximum_compactness << ','
                << field.metric.min_gap << ','
                << initial_lapse.first << ',' << initial_lapse.second << ','
                << radial.r10 << ',' << radial.r50 << ',' << radial.r90
                << ',' << radial.virial_ratio << ','
                << jeans.relative_l1 << '\n';
        write_high_profile(profiles, step, time, grid, field);
        ev::flush_output_checked(history, "high compactness history");
        ev::flush_output_checked(profiles, "high compactness profiles");
    }

    auto save_high_accepted_state = [&](double dt, bool append_diagnostics) {
        write_wpic_restart(restart_path, markers, time, step);
        if (!append_diagnostics) return;
        radial = radial_diagnostics(markers);
        jeans = schwarzschild_jeans_diagnostics(grid, field);
        summary.minimum_r50 = std::min(summary.minimum_r50, radial.r50);
        const auto lapse = lapse_diagnostics(field.metric);
        history << step << ',' << time << ',' << dt << ','
                << field.rest_mass << ',' << field.adm_mass << ','
                << field.maximum_compactness << ',' << field.metric.min_gap << ','
                << lapse.first << ',' << lapse.second << ','
                << radial.r10 << ',' << radial.r50 << ',' << radial.r90 << ','
                << radial.virial_ratio << ',' << jeans.relative_l1 << '\n';
        write_high_profile(profiles, step, time, grid, field);
        ev::flush_output_checked(history, "high compactness history");
        ev::flush_output_checked(profiles, "high compactness profiles");
        std::cout << "  high step=" << step
                  << " t=" << std::setprecision(5) << time
                  << " Cmax=" << field.maximum_compactness
                  << " gap=" << field.metric.min_gap
                  << " r50=" << radial.r50 << '\n';
    };
    // The selector resolves geodesic motion. The coupled step may accept a
    // shorter interval, so advance time by its returned accepted_dt.
    double last_accepted_dt = 0.0;
    while (!options.initialize_only
           && time < options.high_t_end
           && field.metric.min_gap > parameters.gap_stop) {
        double dt = ev::choose_weighted_einstein_vlasov_pic_timestep(
            markers, grid, field.metric, 0.22, options.high_dt_max);
        dt = std::min(dt, options.high_t_end - time);
        ev::WeightedEinsteinVlasovPICStepResult result;
        try {
            result = ev::advance_weighted_einstein_vlasov_pic_self_consistent(
                markers, dt, grid, step_options);
        } catch (const ev::WeightedEinsteinVlasovPICVacuumOuterBoundary& error) {
            summary.schwarzschild_evolution.outer_domain_limited = true;
            summary.schwarzschild_evolution.rejected_trial_dt = dt;
            summary.schwarzschild_evolution.termination_detail = error.what();
            save_high_accepted_state(last_accepted_dt,
                last_csv_step(history_path) < step || last_csv_step(profiles_path) < step);
            break;
        }
        dt = result.accepted_dt;
        field = result.endpoint_field;
        time += dt;
        ++step;
        last_accepted_dt = dt;
        if (step % options.high_output_steps == 0
            || field.metric.min_gap <= parameters.gap_stop
            || time >= options.high_t_end) {
            save_high_accepted_state(dt, true);
        }
    }
    summary.final_min_gap = field.metric.min_gap;
    const auto final_lapse = lapse_diagnostics(field.metric);
    summary.final_min_lapse = final_lapse.first;
    summary.final_maximum_log_redshift = final_lapse.second;
    summary.final_compactness = field.maximum_compactness;
    summary.final_adm_mass = field.adm_mass;
    summary.adm_mass_relative_drift = std::abs(
        summary.final_adm_mass - summary.initial_adm_mass)
        / summary.initial_adm_mass;
    summary.rest_mass_relative_error = std::abs(
        ev::weighted_pic_total_marker_mass(markers)
        - initial_rest_mass) / initial_rest_mass;
    summary.escaped_rest_mass =
        escaped_schwarzschild_rest_weight(markers);
    summary.steps = step;
    summary.final_r50 = radial.r50;
    summary.final_virial_ratio = radial.virial_ratio;
    summary.final_anisotropic_jeans_relative_l1 =
        schwarzschild_jeans_diagnostics(grid, field).relative_l1;
    // This is the Schwarzschild chart's stopping-threshold proxy. Establishing
    // a trapped surface requires the corresponding regular EF calculation.
    summary.horizon_formed =
        field.metric.min_gap <= parameters.gap_stop
        && field.maximum_compactness >= 1.0 - parameters.gap_stop;
    summary.horizon_time = summary.horizon_formed ? time : -1.0;
    summary.first_trapped_time = summary.horizon_time;
    summary.halo_classification = classify_high_halo_history(
        history_path, false, summary, options.high_t_end,
        initial_rest_mass);
    summary.schwarzschild_evolution.final_time = time;
    summary.virialized_halo =
        !summary.schwarzschild_evolution.outer_domain_limited
        && summary.halo_classification.virialized_halo;
    summary.outcome = options.initialize_only
        ? "initialized"
        : (summary.schwarzschild_evolution.outer_domain_limited
        ? "sc_outer_boundary_evolution_limit"
        : (summary.horizon_formed
        ? "black_hole_threshold_proxy"
        : (summary.virialized_halo ? "virialized_halo" : "undetermined")));
    if (!options.export_ef_null_initial_data_path.empty()) {
        // Reattach the independently evolved donor's boundary anchor after
        // the physical future run has finalized all ordinary SC diagnostics.
        summary.null_slice_conserved_adm_mass =
            exported_null_slice_conserved_adm_mass;
    }
    return summary;
}

// Evolve particles on successive ingoing null slices. The reconstructed
// a=1-2m/r can pass through zero while b and P remain positive. The caller
// separately chooses whether to retain all interior particles or excise a
// causally trapped region, and that choice is preserved through restart.
HighSummary run_high_compactness_eddington_finkelstein(
    const RunOptions& options) {
    namespace ef = ev::eddington_finkelstein;
    std::cout << "Running ingoing Eddington--Finkelstein/advanced-null "
              << options.high_initial_condition << " collapse with "
              << options.high_markers << " markers...\n";

    ev::Params parameters = base_parameters(
        options.high_radial_cells,
        options.high_r_min,
        options.high_r_max);
    parameters.gap_stop = options.high_gap_stop;
    parameters.vp_timestep_omitted_mass_fraction =
        options.timestep_omitted_mass_fraction;
    parameters.inner_boundary = options.high_inner_boundary;
    parameters.radial_grid = options.high_radial_grid;
    const ev::WeightedPICFieldGrid grid =
        ef::make_field_grid(parameters, parameters.Nr);

    const std::filesystem::path history_path =
        options.output_dir / "high_compactness_history.csv";
    const std::filesystem::path profiles_path =
        options.output_dir / "high_compactness_profiles.csv";
    const std::filesystem::path restart_path =
        options.output_dir / "high_eddington_finkelstein.restart";
    const std::filesystem::path post_horizon_audit_path =
        ef_post_horizon_audit_path(options.output_dir);

    // Preserve the same excision choice on restart. Removed trajectories
    // remain absent from that state, so changing to a deeper inner boundary
    // requires fresh initial data that still contain those trajectories.
    const auto interior_controls_path = options.output_dir / "ef_interior_controls.txt";
    std::ostringstream interior_controls;
    interior_controls << std::setprecision(17) << "VENUS_EF_INTERIOR_CONTROLS 2\n"
        << options.ef_excision_a << ' ' << options.ef_excision_buffer_cells << ' '
        << options.ef_excision_horizon_fraction << ' '
        << (options.ef_enable_excision ? 1 : 0) << '\n';
    std::ostringstream legacy_interior_controls;
    legacy_interior_controls << std::setprecision(17)
        << "VENUS_EF_INTERIOR_CONTROLS 1\n"
        << options.ef_excision_a << ' ' << options.ef_excision_buffer_cells << ' '
        << options.ef_excision_horizon_fraction << '\n';
    if (options.resume) {
        std::ifstream previous(interior_controls_path);
        std::ostringstream contents;
        contents << previous.rdbuf();
        if (!previous || (contents.str() != interior_controls.str()
            && !(options.ef_enable_excision
                 && contents.str() == legacy_interior_controls.str()))) {
            throw std::runtime_error(
                "EF restart lacks matching interior/excision controls. Start a fresh run; "
                "excised trajectories require initialization from the solved null slice.");
        }
    } else {
        std::ofstream controls_file(interior_controls_path);
        controls_file << interior_controls.str();
        ev::flush_output_checked(controls_file, "EF interior controls");
    }

    std::vector<ef::Marker> markers;
    HighSummary summary;
    EFPostHorizonAuditState post_horizon_audit;
    double advanced_time = 0.0;
    int step = 0;
    double conserved_adm_mass = 0.0;
    double initial_rest_weight = 0.0;
    std::vector<double> mapped_schwarzschild_a;
    std::vector<double> mapped_schwarzschild_b;
    std::vector<ev::coordinate_map::MappedSlicePoint>
        mapped_schwarzschild_slice;

    if (options.resume) {
        const EFRestartState restart = read_ef_restart(restart_path);
        if (restart.enable_excision != options.ef_enable_excision) {
            throw std::runtime_error(
                "EF restart excision mode differs from the requested mode; "
                "start a fresh run from the common null initial data.");
        }
        if (restart.step > std::numeric_limits<int>::max()
            || restart.markers.size() != options.high_markers) {
            throw std::runtime_error(
                "EF restart and configured marker count do not match.");
        }
        markers = restart.markers;
        advanced_time = restart.advanced_time;
        step = static_cast<int>(restart.step);
        conserved_adm_mass = restart.conserved_adm_mass;
        try {
            post_horizon_audit = read_ef_post_horizon_audit(
                post_horizon_audit_path,
                restart.advanced_time,
                restart.step);
        } catch (const std::exception& error) {
            // Checkpoints with incomplete diagnostics may be evolved, but they can
            // never be promoted to intrinsic post-horizon evidence because
            // accepted-step extrema before this resume are unavailable.
            post_horizon_audit.complete = false;
            post_horizon_audit.finite_fields_throughout = false;
            post_horizon_audit.fixed_adm_bookkeeping_throughout = false;
            post_horizon_audit.causal_excision_valid_throughout = false;
            std::cerr << "  warning: " << error.what()
                      << "; post-horizon certification disabled for this "
                         "resumed run.\n";
        }
        const HighResumeHistory accumulated =
            read_ef_high_resume_history(history_path);
        summary = accumulated.summary;
        const FixedFamilyBoundDiagnostics resumed_bound =
            read_fixed_family_bound_diagnostics(options.output_dir);
        validate_energy_cutoff_fixed_family_binding(options, resumed_bound);
        set_fixed_family_bound_diagnostics(summary, resumed_bound);
        summary.null_slice_conserved_adm_mass = conserved_adm_mass;
        initial_rest_weight = accumulated.initial_rest_mass;
        summary.initial_rest_weight = initial_rest_weight;
        if (last_csv_step(history_path) > step
            || last_csv_step(profiles_path) > step) {
            throw std::runtime_error(
                "EF restart is older than existing CSV output.");
        }
        if (!options.physical_ic_path.empty()) {
            if (!std::filesystem::exists(options.physical_ic_path)) {
                throw std::runtime_error(
                    "Configured physical IC is missing during EF resume.");
            }
            auto base_markers = options.high_initial_condition
                    == "energy_cutoff_shell"
                ? load_or_make_high_base_markers(
                    options, parameters, grid, &summary)
                : read_physical_ic(options.physical_ic_path);
            summary.base_ic_fingerprint =
                physical_ic_fingerprint(base_markers);
            apply_high_angular_momentum_controls(base_markers, options);
            const InitialADMNormalization normalization =
                normalize_high_initial_adm_mass(
                    base_markers, options, grid);
            validate_restart_initial_marker_invariants(markers, base_markers);
            summary.initial_rest_weight_scale =
                normalization.rest_weight_scale;
            summary.fixed_family_adm_mass =
                normalization.achieved_adm_mass;
            summary.realized_ic_fingerprint =
                physical_ic_fingerprint(base_markers);
            summary.initial_rest_weight =
                ev::weighted_pic_total_marker_mass(base_markers);
        }
        if (!options.ef_null_initial_data_path.empty()) {
            verify_null_manifest_artifacts(
                options.ef_null_initial_data_path);
            const EFNullInitialData mapped = read_null_initial_data(
                options.ef_null_initial_data_path);
            verify_null_manifest_metadata(
                options.ef_null_initial_data_path, mapped);
            validate_null_initial_data_for_grid(mapped, grid);
            validate_restart_initial_marker_invariants(markers, mapped.markers);
            if (options.physical_ic_path.empty()
                || !std::filesystem::exists(options.physical_ic_path)) {
                throw std::runtime_error(
                    "Mapped EF resume requires its immutable physical IC.");
            }
            auto configured = options.high_initial_condition
                    == "energy_cutoff_shell"
                ? load_or_make_high_base_markers(
                    options, parameters, grid, &summary)
                : read_physical_ic(options.physical_ic_path);
            const std::uint64_t configured_base =
                physical_ic_fingerprint(configured);
            apply_high_angular_momentum_controls(configured, options);
            normalize_high_initial_adm_mass(configured, options, grid);
            const std::uint64_t configured_sc =
                physical_ic_fingerprint(configured);
            if (mapped.markers.size() != restart.markers.size()
                || mapped.conserved_adm_mass != conserved_adm_mass
                || mapped.sampling_realization
                    != options.high_sampling_realization
                || mapped.base_ic_fingerprint != configured_base
                || mapped.schwarzschild_realized_fingerprint
                    != configured_sc) {
                throw std::runtime_error(
                    "EF restart configuration does not match its null-map "
                    "IC, realization, angular controls, or ADM provenance.");
            }
            summary.eikonal_mapped_initial_data = true;
            summary.coordinate_map_manifest_sha256 =
                ev::integrity::sha256_file(null_artifact_path(
                    options.ef_null_initial_data_path,
                    ".manifest.json"));
            summary.eikonal_boundary_radius = mapped.map_worldtube_radius;
            summary.eikonal_evolution_outer_radius = mapped.outer_radius;
            summary.eikonal_v_offset = mapped.ef_v_offset;
            summary.eikonal_analytic_vacuum_extension =
                mapped.analytic_vacuum_extension;
            summary.base_ic_fingerprint = mapped.base_ic_fingerprint;
            summary.realized_ic_fingerprint =
                mapped.ef_realized_fingerprint;
        }
        std::cout << "  resumed EF checkpoint at step=" << step
                  << " v=" << std::setprecision(17) << advanced_time
                  << '\n';
    } else {
        std::vector<ev::WeightedPICMarker> base_markers =
            load_or_make_high_base_markers(
                options, parameters, grid, &summary);
        const std::uint64_t configured_base_fingerprint =
            physical_ic_fingerprint(base_markers);
        apply_high_angular_momentum_controls(base_markers, options);
        const InitialADMNormalization normalization =
            normalize_high_initial_adm_mass(base_markers, options, grid);
        const std::uint64_t configured_sc_fingerprint =
            physical_ic_fingerprint(base_markers);
        const ev::WeightedEinsteinVlasovPICField family_field =
            ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
                base_markers,
                grid,
                ev::WeightedPICParticleShape::tsc,
                options.threads);
        const FixedFamilyBoundDiagnostics bound =
            fixed_family_bound_diagnostics(
                base_markers, grid, family_field);
        validate_energy_cutoff_fixed_family_binding(options, bound);
        set_fixed_family_bound_diagnostics(summary, bound);
        write_fixed_family_bound_diagnostics(options.output_dir, bound);
        verify_null_manifest_artifacts(options.ef_null_initial_data_path);
        const EFNullInitialData mapped = read_null_initial_data(
            options.ef_null_initial_data_path);
        verify_null_manifest_metadata(
            options.ef_null_initial_data_path, mapped);
        validate_null_initial_data_for_grid(mapped, grid);
        if (mapped.markers.size() != options.high_markers
            || mapped.base_ic_fingerprint != configured_base_fingerprint
            || mapped.schwarzschild_realized_fingerprint
                != configured_sc_fingerprint
            || mapped.sampling_realization
                != options.high_sampling_realization) {
            throw std::runtime_error(
                "Solved-eikonal data do not match the configured physical "
                "IC, angular controls, marker count, or radial grid.");
        }
        // These momenta and weights have already been transformed onto the
        // selected null slice. Use that slice's mass as the isolated outer
        // constraint, retaining the initial Schwarzschild family mass separately.
        markers = mapped.markers;
        conserved_adm_mass = mapped.conserved_adm_mass;
        initial_rest_weight = 0.0;
        for (const auto& marker : markers) {
            initial_rest_weight += marker.rest_weight;
        }
        summary.base_ic_fingerprint = mapped.base_ic_fingerprint;
        summary.realized_ic_fingerprint = mapped.ef_realized_fingerprint;
        summary.initial_rest_weight_scale =
            normalization.rest_weight_scale;
        summary.fixed_family_adm_mass =
            normalization.achieved_adm_mass;
        summary.initial_rest_weight = initial_rest_weight;
        summary.eikonal_mapped_initial_data = true;
        summary.coordinate_map_manifest_sha256 =
            ev::integrity::sha256_file(null_artifact_path(
                options.ef_null_initial_data_path, ".manifest.json"));
        summary.eikonal_boundary_radius = mapped.map_worldtube_radius;
        summary.eikonal_evolution_outer_radius = mapped.outer_radius;
        summary.eikonal_v_offset = mapped.ef_v_offset;
        summary.eikonal_analytic_vacuum_extension =
            mapped.analytic_vacuum_extension;
        summary.null_slice_conserved_adm_mass =
            mapped.conserved_adm_mass;
        mapped_schwarzschild_a.resize(grid.radius.size());
        mapped_schwarzschild_b.resize(grid.radius.size());
        mapped_schwarzschild_slice = mapped.mapped_slice;
        for (std::size_t i = 0; i < grid.radius.size(); ++i) {
            mapped_schwarzschild_a[i] = mapped.mapped_slice[i].a;
            mapped_schwarzschild_b[i] = mapped.mapped_slice[i].b;
        }
    }

    // The finite outer metric is matched to an isolated vacuum exterior.
    // Escaped gravitating matter requires an exterior model before resume.
    ef::check_vacuum_outer_boundary(markers);
    ef::ConstraintOptions constraints;
    constraints.excision_a_threshold = options.ef_excision_a;
    constraints.excision_buffer_cells = options.ef_excision_buffer_cells;
    constraints.excision_horizon_fraction = options.ef_excision_horizon_fraction;
    constraints.enable_excision = options.ef_enable_excision;
    if (options.resume || !options.ef_null_initial_data_path.empty()) {
        constraints.fixed_adm_mass = conserved_adm_mass;
    }
    ef::Field field = ef::deposit_and_solve_field(
        markers,
        grid,
        ev::WeightedPICParticleShape::tsc,
        options.threads,
        constraints);
    if (!options.resume && options.ef_null_initial_data_path.empty()) {
        throw std::runtime_error(
            "Internal error: fresh EF field lacks null-map provenance.");
    }
    if (!options.resume) {
        if (mapped_schwarzschild_slice.size() != grid.radius.size()) {
            throw std::runtime_error(
                "Fresh EF data lack a complete mapped tensor slice.");
        }
        const EFCovariantStressAudit stress =
            deposit_ef_covariant_stress_audit(
                markers, grid, field.metric);
        long double t_rr_error = 0.0L;
        long double t_rr_reference = 0.0L;
        long double s_error = 0.0L;
        long double s_reference = 0.0L;
        long double t_vv_error = 0.0L;
        long double t_vv_reference = 0.0L;
        long double t_vr_error = 0.0L;
        long double t_vr_reference = 0.0L;
        long double t_theta_error = 0.0L;
        long double t_theta_reference = 0.0L;
        std::ofstream audit(
            options.output_dir / "initial_coordinate_map_residuals.csv");
        if (!audit) {
            throw std::runtime_error(
                "Could not create initial coordinate-map residual audit.");
        }
        audit << std::setprecision(17)
            << "r,mapped_a,ef_a,delta_a,mapped_b,ef_b,delta_b,"
               "mapped_m,ef_m,delta_m,mapped_T_rr,ef_T_rr,delta_T_rr,"
               "mapped_S,ef_S,delta_S,mapped_T_vv,ef_T_vv,delta_T_vv,"
               "mapped_T_vr,ef_T_vr,delta_T_vr,mapped_T_theta_theta,"
               "ef_T_theta_theta,delta_T_theta_theta\n";
        for (std::size_t i = 0; i < grid.radius.size(); ++i) {
            const auto& mapped = mapped_schwarzschild_slice[i];
            const long double volume = grid.mesh.shell_volume[i];
            auto accumulate = [&](double actual, double expected,
                                  long double& error,
                                  long double& reference) {
                const long double difference = actual - expected;
                error += volume * difference * difference;
                reference += volume * expected * expected;
            };
            summary.initial_schwarzschild_map_a_linf = std::max(
                summary.initial_schwarzschild_map_a_linf,
                std::abs(field.metric.a[i] - mapped.a));
            summary.initial_schwarzschild_map_b_linf = std::max(
                summary.initial_schwarzschild_map_b_linf,
                std::abs(field.metric.b[i] - mapped.b));
            summary.initial_schwarzschild_map_m_linf = std::max(
                summary.initial_schwarzschild_map_m_linf,
                std::abs(field.metric.m[i] - mapped.m));
            accumulate(
                field.sources.t_rr[i], mapped.t_rr,
                t_rr_error, t_rr_reference);
            accumulate(
                field.sources.s[i], mapped.s, s_error, s_reference);
            accumulate(
                stress.t_vv[i], mapped.t_vv,
                t_vv_error, t_vv_reference);
            accumulate(
                stress.t_vr[i], mapped.t_vr,
                t_vr_error, t_vr_reference);
            accumulate(
                stress.t_theta_theta[i], mapped.t_theta_theta,
                t_theta_error, t_theta_reference);
            audit << grid.radius[i] << ','
                  << mapped.a << ',' << field.metric.a[i] << ','
                  << field.metric.a[i] - mapped.a << ','
                  << mapped.b << ',' << field.metric.b[i] << ','
                  << field.metric.b[i] - mapped.b << ','
                  << mapped.m << ',' << field.metric.m[i] << ','
                  << field.metric.m[i] - mapped.m << ','
                  << mapped.t_rr << ',' << field.sources.t_rr[i] << ','
                  << field.sources.t_rr[i] - mapped.t_rr << ','
                  << mapped.s << ',' << field.sources.s[i] << ','
                  << field.sources.s[i] - mapped.s << ','
                  << mapped.t_vv << ',' << stress.t_vv[i] << ','
                  << stress.t_vv[i] - mapped.t_vv << ','
                  << mapped.t_vr << ',' << stress.t_vr[i] << ','
                  << stress.t_vr[i] - mapped.t_vr << ','
                  << mapped.t_theta_theta << ','
                  << stress.t_theta_theta[i] << ','
                  << stress.t_theta_theta[i] - mapped.t_theta_theta
                  << '\n';
        }
        auto relative_l2 = [](long double error, long double reference) {
            return std::sqrt(static_cast<double>(
                error / std::max(reference, 1.0e-300L)));
        };
        summary.initial_schwarzschild_map_t_rr_relative_l2 =
            relative_l2(t_rr_error, t_rr_reference);
        summary.initial_schwarzschild_map_s_relative_l2 =
            relative_l2(s_error, s_reference);
        summary.initial_schwarzschild_map_t_vv_relative_l2 =
            relative_l2(t_vv_error, t_vv_reference);
        summary.initial_schwarzschild_map_t_vr_relative_l2 =
            relative_l2(t_vr_error, t_vr_reference);
        summary.initial_schwarzschild_map_t_theta_theta_relative_l2 =
            relative_l2(t_theta_error, t_theta_reference);
        if (!std::isfinite(
                summary.initial_schwarzschild_map_t_rr_relative_l2)
            || !std::isfinite(
                summary.initial_schwarzschild_map_s_relative_l2)
            || summary.initial_schwarzschild_map_a_linf > 0.35
            || summary.initial_schwarzschild_map_b_linf > 0.35
            || summary.initial_schwarzschild_map_t_rr_relative_l2 > 0.75
            || summary.initial_schwarzschild_map_s_relative_l2 > 0.75) {
            throw std::runtime_error(
                "Mapped SC and reconstructed EF metric/stress data fail "
                "the coarse initial parity guard; refine the donor/grid.");
        }
    }

    ef::EvolutionOptions evolution;
    evolution.shape = ev::WeightedPICParticleShape::tsc;
    evolution.threads = options.threads;
    evolution.characteristic_substeps = 2;
    evolution.fixed_adm_mass = conserved_adm_mass;
    evolution.enable_excision = options.ef_enable_excision;
    evolution.require_vacuum_outer_boundary = true;
    evolution.excision_a_threshold = options.ef_excision_a;
    evolution.excision_buffer_cells = options.ef_excision_buffer_cells;
    evolution.excision_horizon_fraction = options.ef_excision_horizon_fraction;

    RadialDiagnostics radial =
        ef_radial_diagnostics(markers, grid, field.metric);
    const AngularMomentumDiagnostics angular =
        angular_momentum_diagnostics(markers);
    const double minimum_b = *std::min_element(
        field.metric.b.begin(), field.metric.b.end());
    if (!options.resume) {
        summary.initial_min_gap = field.metric.minimum_a;
        summary.initial_min_b = minimum_b;
        summary.initial_min_lapse = -1.0;
        summary.initial_compactness = field.metric.maximum_compactness;
        summary.initial_deposited_adm_mass = field.metric.adm_mass;
        summary.initial_adm_mass = summary.initial_deposited_adm_mass;
        summary.null_slice_conserved_adm_mass = conserved_adm_mass;
        summary.initial_r50 = radial.r50;
        summary.minimum_r50 = radial.r50;
        if (field.metric.apparent_horizon_found) {
            summary.horizon_formed = true;
            summary.horizon_time = advanced_time;
            summary.first_trapped_time = advanced_time;
            summary.apparent_horizon_radius =
                field.metric.apparent_horizon_radius;
        }
        if (!mapped_schwarzschild_a.empty()) {
            for (std::size_t i = 0; i < grid.radius.size(); ++i) {
                summary.initial_schwarzschild_map_a_linf = std::max(
                    summary.initial_schwarzschild_map_a_linf,
                    std::abs(field.metric.a[i]
                        - mapped_schwarzschild_a[i]));
                summary.initial_schwarzschild_map_b_linf = std::max(
                    summary.initial_schwarzschild_map_b_linf,
                    std::abs(field.metric.b[i]
                        - mapped_schwarzschild_b[i]));
            }
        }
    }
    update_ef_post_horizon_audit(
        post_horizon_audit,
        markers,
        field,
        conserved_adm_mass,
        false,
        evolution.excision_a_threshold,
        evolution.excision_buffer_cells);
    if (!options.resume) {
        write_ef_restart(
            restart_path,
            {markers, advanced_time, step, conserved_adm_mass,
             options.ef_enable_excision});
        write_ef_post_horizon_audit(
            post_horizon_audit_path,
            post_horizon_audit,
            advanced_time,
            step);
    }
    summary.mass_weighted_mean_specific_angular_momentum = angular.mean;
    summary.mass_weighted_rms_specific_angular_momentum = angular.rms;
    summary.maximum_specific_angular_momentum = angular.maximum;

    std::ofstream history = open_csv_output(
        history_path,
        options.resume,
        "step,advanced_time,dt,total_rest_weight,active_rest_weight,"
        "excised_rest_weight,escaped_rest_weight,adm_mass,max_compactness,"
        "min_a,min_b,apparent_horizon_radius,r10,r50,r90,virial_ratio,"
        "excision_radius,anisotropic_jeans_relative_l1,active_marker_count,"
        "active_trapped_rest_weight,active_min_radius",
        "EF high-compactness history.csv");
    std::ofstream profiles = open_csv_output(
        profiles_path,
        options.resume,
        "step,advanced_time,r,a,b,m,compactness,T_rr,S,T_vv,T_vr,"
        "T_theta_theta,rest_weight_density,metric_sample_kind,metric_cell,"
        "face_left_r,face_right_r,face_left_a,face_right_a,face_left_b,"
        "face_right_b,rho_infall,excision_mask",
        "EF high-compactness profiles.csv");

    auto write_history_row = [&](double dt) {
        const double horizon_radius = field.metric.apparent_horizon_found
            ? field.metric.apparent_horizon_radius : -1.0;
        const double min_b = *std::min_element(
            field.metric.b.begin(), field.metric.b.end());
        const AnisotropicJeansDiagnostics jeans =
            ef_jeans_diagnostics(markers, grid, field);
        std::size_t active_count = 0;
        long double trapped_weight = 0.0L;
        double active_min_radius = std::numeric_limits<double>::infinity();
        const ef::MetricSampler metric_sampler(grid, field.metric);
        for (const auto& marker : markers) {
            if (marker.state != ef::MarkerState::active) continue;
            ++active_count;
            active_min_radius = std::min(active_min_radius, marker.r);
            if (metric_sampler(marker.r).a < 0.0) {
                trapped_weight += marker.rest_weight;
            }
        }
        history << step << ',' << advanced_time << ',' << dt << ','
                << field.total_rest_weight << ','
                << field.sources.active_rest_weight << ','
                << field.sources.excised_rest_weight << ','
                << field.sources.escaped_rest_weight << ','
                << field.metric.adm_mass << ','
                << field.metric.maximum_compactness << ','
                << field.metric.minimum_a << ',' << min_b << ','
                << horizon_radius << ',' << radial.r10 << ','
                << radial.r50 << ',' << radial.r90 << ','
                << radial.virial_ratio << ','
                << field.metric.excision_radius << ','
                << jeans.relative_l1 << ',' << active_count << ','
                << static_cast<double>(trapped_weight) << ','
                << (active_count ? active_min_radius : -1.0) << '\n';
    };

    const bool checkpoint_already_in_output = options.resume
        && last_csv_step(history_path) == step
        && last_csv_step(profiles_path) == step;
    if (!checkpoint_already_in_output) {
        write_history_row(0.0);
        write_ef_high_profile(
            profiles, step, advanced_time, markers, grid, field);
        ev::flush_output_checked(history, "EF high compactness history");
        ev::flush_output_checked(profiles, "EF high compactness profiles");
    }

    std::int64_t last_saved_step = step;
    auto record_evolution_limit = [&](const std::string& reason,
                                       const std::string& detail,
                                       double trial_dt) {
        summary.ef_termination_reason = reason;
        summary.ef_termination_detail = detail;
        summary.ef_final_trial_dt = trial_dt;
    };
    while (!options.initialize_only
           && advanced_time < options.high_t_end) {
        double dt = 0.0;
        try {
            dt = ef::choose_timestep(
                markers, grid, field.metric, 0.20, options.high_dt_max);
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            if (options.ef_enable_excision
                || (message.find("EF WPIC timestep estimate") == std::string::npos
                    && message.find("EF characteristic RHS") == std::string::npos)) {
                throw;
            }
            record_evolution_limit("interior_timestep_limit", message, dt);
            break;
        }
        // Land on the requested advanced time. A positive floating-point dt
        // must also produce a representably later coordinate time.
        dt = std::min(dt, options.high_t_end - advanced_time);
        if (!(dt > 0.0) || !(advanced_time + dt > advanced_time)) {
            if (options.ef_enable_excision) {
                throw std::runtime_error("EF timestep cannot advance representable time.");
            }
            record_evolution_limit("interior_time_resolution_limit",
                "The selected timestep cannot advance representable advanced time.", dt);
            break;
        }
        ef::StepResult result;
        int characteristic_bisections = 0;
        bool accepted = false;
        // Retry only recognized characteristic or interior trial failures.
        // The solver preserves the incoming state on rejection, allowing a
        // smaller interval to start from the same particles and metric.
        for (;;) {
            try {
                result = ef::advance_self_consistent(
                    markers, dt, grid, evolution);
                accepted = true;
                break;
            } catch (const ef::VacuumOuterBoundaryViolation& error) {
                record_evolution_limit("vacuum_outer_boundary_limit", error.what(), dt);
                break;
            } catch (const std::runtime_error& error) {
                const std::string message = error.what();
                const bool retryable_interior_constraint = !options.ef_enable_excision
                    && (message.find("EF metric constraint produced non-finite data")
                            != std::string::npos
                        || message.find("EF metric solve received invalid sources/options")
                            != std::string::npos);
                const bool retryable_trial_failure =
                    message.find("EF characteristic RHS")
                        != std::string::npos
                    || message.find("EF RK4") != std::string::npos
                    || message.find("EF log-momentum")
                        != std::string::npos
                    || (!options.ef_enable_excision
                        && message.find("An interior EF marker crossed the inner field boundary")
                            != std::string::npos)
                    || retryable_interior_constraint;
                const bool can_bisect = characteristic_bisections < 24
                    && (options.ef_enable_excision
                        ? 0.5 * dt > 64.0 * std::numeric_limits<double>::epsilon()
                            * std::max(1.0, advanced_time)
                        : advanced_time + 0.5 * dt > advanced_time);
                if (!retryable_trial_failure
                    || !can_bisect) {
                    const bool constraint_failure = message.find(
                        "EF metric constraint produced non-finite data") != std::string::npos
                        || message.find("EF metric solve received invalid sources/options")
                            != std::string::npos;
                    const bool inner_failure = message.find(
                        "An interior EF marker crossed the inner field boundary")
                            != std::string::npos;
                    if (!options.ef_enable_excision
                        && (retryable_trial_failure || constraint_failure || inner_failure)) {
                        record_evolution_limit(
                            constraint_failure ? "interior_constraint_limit"
                                : (inner_failure ? "interior_inner_boundary_limit"
                                                 : "interior_characteristic_limit"),
                            message, dt);
                        break;
                    }
                    throw std::runtime_error(
                        message
                        + " The advanced-null chart does not support a "
                          "pre-horizon particle crossing r=0; otherwise "
                          "reduce --high-dt-max or increase characteristic "
                          "resolution.");
                }
                // Halve the numerical interval while keeping the physical
                // state, domain and error checks unchanged. The finite retry
                // cap and time-resolution test above bound this refinement.
                dt *= 0.5;
                ++characteristic_bisections;
            }
        }
        if (!accepted) {
            break;
        }
        field = result.endpoint_field;
        advanced_time += result.accepted_dt;
        summary.ef_last_accepted_dt = result.accepted_dt;
        ++step;

        // Diagnose trapping from the solved field. Record the first accepted
        // slice with an apparent-horizon crossing and force an output there.
        bool horizon_just_formed = false;
        if (field.metric.apparent_horizon_found) {
            if (!summary.horizon_formed) {
                summary.horizon_formed = true;
                summary.horizon_time = advanced_time;
                summary.first_trapped_time = advanced_time;
                horizon_just_formed = true;
            }
            summary.apparent_horizon_radius =
                field.metric.apparent_horizon_radius;
        }
        update_ef_post_horizon_audit(
            post_horizon_audit,
            markers,
            field,
            conserved_adm_mass,
            true,
            evolution.excision_a_threshold,
            evolution.excision_buffer_cells);
        if (step % options.high_output_steps == 0
            || advanced_time >= options.high_t_end
            || horizon_just_formed) {
            radial = ef_radial_diagnostics(markers, grid, field.metric);
            summary.minimum_r50 = std::min(
                summary.minimum_r50, radial.r50);
            write_ef_restart(
                restart_path,
                {markers, advanced_time, step, conserved_adm_mass,
                 options.ef_enable_excision});
            write_ef_post_horizon_audit(
                post_horizon_audit_path,
                post_horizon_audit,
                advanced_time,
                step);
            write_history_row(result.accepted_dt);
            write_ef_high_profile(
                profiles, step, advanced_time, markers, grid, field);
            ev::flush_output_checked(
                history, "EF high compactness history");
            ev::flush_output_checked(
                profiles, "EF high compactness profiles");
            last_saved_step = step;
            std::cout << "  EF step=" << step
                      << " v=" << std::setprecision(6) << advanced_time
                      << " Cmax=" << field.metric.maximum_compactness
                      << " min(a)=" << field.metric.minimum_a
                      << " r_AH="
                      << (field.metric.apparent_horizon_found
                          ? field.metric.apparent_horizon_radius : -1.0)
                      << " M_exc=" << field.sources.excised_rest_weight
                      << '\n';
        }
    }

    // A rejected limited-evolution trial leaves markers and field at the last
    // accepted slice. Save that complete slice even between output cadences.
    if (!summary.ef_termination_reason.empty()) {
        radial = ef_radial_diagnostics(markers, grid, field.metric);
        write_ef_restart(restart_path,
            {markers, advanced_time, step, conserved_adm_mass,
             options.ef_enable_excision});
        write_ef_post_horizon_audit(
            post_horizon_audit_path, post_horizon_audit, advanced_time, step);
        if (last_saved_step != step) {
            write_history_row(summary.ef_last_accepted_dt);
            write_ef_high_profile(
                profiles, step, advanced_time, markers, grid, field);
        }
        ev::flush_output_checked(history, "EF limited-evolution history");
        ev::flush_output_checked(profiles, "EF limited-evolution profiles");
        std::cout << "  EF limited evolution at v=" << std::setprecision(17)
                  << advanced_time << ": " << summary.ef_termination_reason
                  << "; " << summary.ef_termination_detail << '\n';
    } else {
        summary.ef_termination_reason = options.initialize_only
            ? "initialized" : "requested_end_time";
    }
    summary.ef_final_advanced_time = advanced_time;
    summary.ef_requested_end_time_reached = advanced_time >= options.high_t_end;
    summary.ef_first_field_center_radius = grid.radius.front();
    summary.ef_first_cell_outer_radius = grid.mesh.edge[1];
    summary.ef_inner_face_mass = field.metric.face_m.front();
    summary.ef_inner_face_mb = field.metric.face_mb.front();
    summary.ef_inner_face_b = field.metric.face_b.front();
    // Measure how much retained matter lies below the innermost deposition
    // scale. These diagnostics distinguish an unresolved central population
    // from an actual trapped surface found in the metric.
    double active_minimum = std::numeric_limits<double>::infinity();
    ev::weighted_pic_detail::CompensatedMassSum inner_center_weight;
    ev::weighted_pic_detail::CompensatedMassSum inner_cell_weight;
    for (const auto& marker : markers) {
        if (marker.state != ef::MarkerState::active) continue;
        active_minimum = std::min(active_minimum, marker.r);
        if (marker.r < grid.radius.front()) {
            ++summary.ef_active_markers_below_first_field_center;
            inner_center_weight.add(marker.rest_weight);
        }
        if (marker.r < grid.mesh.edge[1]) {
            inner_cell_weight.add(marker.rest_weight);
        }
    }
    summary.ef_final_active_min_radius = std::isfinite(active_minimum)
        ? active_minimum : -1.0;
    summary.ef_active_weight_inside_first_center =
        static_cast<double>(inner_center_weight.value());
    summary.ef_active_weight_inside_first_cell =
        static_cast<double>(inner_cell_weight.value());
    record_ef_spatial_resolution(
        summary, markers, grid, field, options.ef_enable_excision);
    radial = ef_radial_diagnostics(markers, grid, field.metric);
    summary.final_min_gap = field.metric.minimum_a;
    summary.final_min_b = *std::min_element(
        field.metric.b.begin(), field.metric.b.end());
    summary.final_min_lapse = -1.0;
    summary.final_maximum_log_redshift = -1.0;
    summary.final_compactness = field.metric.maximum_compactness;
    summary.final_adm_mass = field.metric.adm_mass;
    summary.adm_mass_relative_drift = std::abs(
        summary.final_adm_mass - summary.initial_adm_mass)
        / std::max(summary.initial_adm_mass, 1.0e-300);
    summary.rest_mass_relative_error = std::abs(
        field.total_rest_weight - initial_rest_weight)
        / std::max(initial_rest_weight, 1.0e-300);
    summary.excised_rest_mass = field.sources.excised_rest_weight;
    summary.escaped_rest_mass = field.sources.escaped_rest_weight;
    summary.steps = step;
    summary.final_r50 = radial.r50;
    summary.final_virial_ratio = radial.virial_ratio;
    summary.final_anisotropic_jeans_relative_l1 =
        ef_jeans_diagnostics(markers, grid, field).relative_l1;
    summary.evolved_past_horizon = summary.horizon_formed
        && advanced_time > summary.horizon_time;
    summary.post_horizon_advanced_time = summary.horizon_formed
        ? std::max(0.0, advanced_time - summary.horizon_time) : 0.0;
    copy_ef_post_horizon_audit_to_summary(
        post_horizon_audit, summary);
    summary.halo_classification = classify_high_halo_history(
        history_path, true, summary, options.high_t_end,
        initial_rest_weight);
    summary.virialized_halo =
        summary.halo_classification.virialized_halo;
    const bool limited_evolution = summary.ef_termination_reason != "requested_end_time"
        && summary.ef_termination_reason != "initialized";
    summary.outcome = limited_evolution
        ? (summary.ef_termination_reason == "vacuum_outer_boundary_limit"
            ? "ef_outer_boundary_evolution_limit" : "interior_evolution_limit")
        : options.initialize_only
        ? "initialized"
        : (summary.horizon_formed
            ? "black_hole"
            : (summary.virialized_halo
                ? "virialized_halo" : "undetermined"));
    return summary;
}

// ---- Machine-readable run summary ----------------------------------------
// Record each physical check used by the classifier together with its
// threshold. A stable late-time radius alone does not establish equilibrium.
void write_halo_classification_json(
    std::ostream& output,
    const ev::EinsteinVlasovHaloClassification& value) {
    const auto& threshold = value.thresholds;
    output
        << "{\n"
        << "      \"schema\": \"venus_wpic_halo_classifier_v1\",\n"
        << "      \"virialized_halo\": "
        << (value.virialized_halo ? "true" : "false") << ",\n"
        << "      \"history_valid\": "
        << (value.history_valid ? "true" : "false") << ",\n"
        << "      \"completed_end_time\": "
        << (value.completed_end_time ? "true" : "false") << ",\n"
        << "      \"positive_binding_energy\": "
        << (value.positive_binding_energy ? "true" : "false") << ",\n"
        << "      \"no_horizon\": "
        << (value.no_horizon ? "true" : "false") << ",\n"
        << "      \"escape_small\": "
        << (value.escape_small ? "true" : "false") << ",\n"
        << "      \"compression_occurred\": "
        << (value.compression_occurred ? "true" : "false") << ",\n"
        << "      \"late_series_available\": "
        << (value.late_series_available ? "true" : "false") << ",\n"
        << "      \"not_near_horizon_collapse\": "
        << (value.not_near_horizon_collapse ? "true" : "false") << ",\n"
        << "      \"history_samples\": " << value.history_samples << ",\n"
        << "      \"late_samples\": " << value.late_samples << ",\n"
        << "      \"late_anisotropic_jeans_samples\": "
        << value.late_jeans_samples << ",\n"
        << "      \"minimum_late_anisotropic_jeans_samples\": "
        << value.minimum_late_jeans_samples << ",\n"
        << "      \"final_time\": " << value.final_time << ",\n"
        << "      \"minimum_r50\": " << value.minimum_r50 << ",\n"
        << "      \"maximum_history_compactness\": "
        << value.maximum_history_compactness << ",\n"
        << "      \"escaped_rest_fraction\": "
        << value.escaped_rest_fraction << ",\n"
        << "      \"late_r50_median\": " << value.late_r50_median << ",\n"
        << "      \"late_r50_relative_spread\": "
        << value.late_r50_relative_spread << ",\n"
        << "      \"late_r50_dimensionless_trend\": "
        << value.late_r50_dimensionless_trend << ",\n"
        << "      \"late_r90_dimensionless_trend\": "
        << value.late_r90_dimensionless_trend << ",\n"
        << "      \"late_compactness_absolute_spread\": "
        << value.late_compactness_absolute_spread << ",\n"
        << "      \"late_compactness_absolute_trend\": "
        << value.late_compactness_absolute_trend << ",\n"
        << "      \"late_anisotropic_jeans_relative_l1_median\": "
        << value.late_anisotropic_jeans_relative_l1_median << ",\n"
        << "      \"late_anisotropic_jeans_relative_l1_p90\": "
        << value.late_anisotropic_jeans_relative_l1_p90 << ",\n"
        << "      \"late_anisotropic_jeans_absolute_trend\": "
        << value.late_anisotropic_jeans_absolute_trend << ",\n"
        << "      \"thresholds\": {\n"
        << "        \"end_time_relative_tolerance\": "
        << threshold.end_time_relative_tolerance << ",\n"
        << "        \"late_window_fraction\": "
        << threshold.late_window_fraction << ",\n"
        << "        \"minimum_late_samples\": "
        << threshold.minimum_late_samples << ",\n"
        << "        \"compression_fraction\": "
        << threshold.compression_fraction << ",\n"
        << "        \"maximum_late_r50_over_initial\": "
        << threshold.maximum_late_r50_over_initial << ",\n"
        << "        \"maximum_r50_relative_spread\": "
        << threshold.maximum_r50_relative_spread << ",\n"
        << "        \"maximum_dimensionless_r50_trend\": "
        << threshold.maximum_dimensionless_r50_trend << ",\n"
        << "        \"maximum_dimensionless_r90_trend\": "
        << threshold.maximum_dimensionless_r90_trend << ",\n"
        << "        \"maximum_compactness_absolute_spread\": "
        << threshold.maximum_compactness_absolute_spread << ",\n"
        << "        \"maximum_compactness_absolute_trend\": "
        << threshold.maximum_compactness_absolute_trend << ",\n"
        << "        \"maximum_anisotropic_jeans_relative_l1\": "
        << threshold.maximum_anisotropic_jeans_relative_l1 << ",\n"
        << "        \"maximum_anisotropic_jeans_late_increase\": "
        << threshold.maximum_anisotropic_jeans_late_increase << ",\n"
        << "        \"minimum_late_jeans_fraction\": "
        << threshold.minimum_late_jeans_fraction << ",\n"
        << "        \"maximum_escaped_rest_fraction\": "
        << threshold.maximum_escaped_rest_fraction << ",\n"
        << "        \"near_horizon_compactness\": "
        << threshold.near_horizon_compactness << "\n"
        << "      },\n"
        << "      \"failed_checks\": [";
    for (std::size_t i = 0; i < value.failed_checks.size(); ++i) {
        output << (i == 0 ? "" : ", ")
               << '"' << value.failed_checks[i] << '"';
    }
    output << "]\n    }";
}

void write_schwarzschild_evolution_status(
    std::ostream& output,
    const SchwarzschildEvolutionStatus& status,
    double requested_end,
    bool available = true) {
    output << "{\n"
        << "      \"available\": " << (available ? "true" : "false") << ",\n"
        << "      \"require_vacuum_outer_boundary\": true,\n"
        << "      \"requested_end_time_reached\": "
        << (available && !status.outer_domain_limited && status.final_time >= requested_end
                ? "true" : "false") << ",\n"
        << "      \"final_time\": " << status.final_time << ",\n"
        << "      \"termination_reason\": \""
        << (status.outer_domain_limited ? "vacuum_outer_boundary_limit" : "") << "\",\n"
        << "      \"termination_detail\": \""
        << json_escape(status.termination_detail) << "\",\n"
        << "      \"rejected_trial_dt\": " << status.rejected_trial_dt << "\n"
        << "    }";
}

void write_summary(
    const RunOptions& options,
    const LowSummary* low,
    const HighSummary* high) {
    std::ofstream output(options.output_dir / "summary.json");
    if (!output) {
        throw std::runtime_error("Could not create summary.json.");
    }
    output << std::setprecision(17);
    output << "{\n"
           << "  \"module\": \"weighted_einstein_vlasov_pic\",\n"
           << "  \"coordinates\": \"" << options.coordinates << "\",\n"
           << "  \"expected_outcome\": \""
           << options.expected_outcome << "\",\n"
           << "  \"collision_operator\": \"none\",\n"
           << "  \"background\": \"static\",\n"
           << "  \"quick\": " << (options.quick ? "true" : "false");
    if (low != nullptr) {
        output << ",\n  \"low_compactness\": {\n"
               << "    \"schwarzschild_evolution_status\": ";
        write_schwarzschild_evolution_status(output,
            low->schwarzschild_evolution, options.low_t_end);
        output
            << ",\n    \"markers\": " << options.low_markers << ",\n"
            << "    \"radial_cells\": " << options.low_radial_cells << ",\n"
            << "    \"end_time\": " << options.low_t_end << ",\n"
            << "    \"cfl\": " << options.low_cfl << ",\n"
            << "    \"total_marker_mass\": "
            << options.low_total_mass << ",\n"
            << "    \"initial_velocity_factor\": "
            << options.low_velocity_factor << ",\n"
            << "    \"radial_importance_exponent\": "
            << options.low_radial_importance_exponent << ",\n"
            << "    \"timestep_omitted_mass_fraction\": "
            << options.timestep_omitted_mass_fraction << ",\n"
            << "    \"initial_max_compactness\": "
            << low->initial_compactness << ",\n"
            << "    \"maximum_compactness\": "
            << low->maximum_compactness << ",\n"
            << "    \"maximum_compactness_time\": "
            << low->maximum_compactness_time << ",\n"
            << "    \"final_max_compactness\": "
            << low->final_compactness << ",\n"
            << "    \"minimum_metric_gap\": "
            << low->minimum_metric_gap << ",\n"
            << "    \"maximum_adm_mass_relative_drift\": "
            << low->maximum_adm_mass_relative_drift << ",\n"
            << "    \"initial_virial_ratio\": "
            << low->initial_virial_ratio << ",\n"
            << "    \"final_virial_ratio\": "
            << low->final_virial_ratio << ",\n"
            << "    \"initial_r50\": " << low->initial_r50 << ",\n"
            << "    \"minimum_r50\": " << low->minimum_r50 << ",\n"
            << "    \"final_r50\": " << low->final_r50 << ",\n"
            << "    \"ev_vp_density_l1_difference\": "
            << low->density_l1_difference << ",\n"
            << "    \"ev_vp_particle_rms_radius_difference\": "
            << low->particle_rms_radius_difference << ",\n"
            << "    \"ev_vp_particle_rms_momentum_difference\": "
            << low->particle_rms_momentum_difference << ",\n"
            << "    \"ev_rest_mass_relative_error\": "
            << low->ev_rest_mass_relative_error << ",\n"
            << "    \"vp_rest_mass_relative_error\": "
            << low->vp_rest_mass_relative_error << ",\n"
            << "    \"virialized_halo\": "
            << (low->virialized_halo ? "true" : "false") << ",\n"
            << "    \"agrees_with_vlasov_poisson\": "
            << (low->agrees_with_vlasov_poisson ? "true" : "false")
            << "\n  }";
    }
    if (high != nullptr) {
        output << ",\n  \"high_compactness\": {\n"
               << "    \"schwarzschild_evolution_status\": ";
        write_schwarzschild_evolution_status(output,
            high->schwarzschild_evolution, options.high_t_end,
            options.coordinates == "schwarzschild");
        output
            << ",\n    \"initial_condition\": \""
            << options.high_initial_condition << "\",\n"
            << "    \"coordinates\": \""
            << options.coordinates << "\",\n"
            << "    \"time_coordinate\": \""
            << (options.coordinates == "eddington_finkelstein"
                ? "advanced_null_v" : "polar_areal_t") << "\",\n"
            << "    \"initial_hypersurface\": \""
            << (options.coordinates == "eddington_finkelstein"
                ? "v=0 ingoing-null hypersurface"
                : "t=0 spacelike polar-areal hypersurface") << "\",\n"
            << "    \"eikonal_mapped_initial_data\": "
            << (high->eikonal_mapped_initial_data ? "true" : "false")
            << ",\n"
            << "    \"coordinate_map_role\": \""
            << (options.coordinates == "eddington_finkelstein"
                ? "mapped_null_initial_data_consumer"
                : (high->eikonal_mapped_initial_data
                    ? "time_reversed_schwarzschild_donor"
                    : "none")) << "\",\n"
            << "    \"cross_coordinate_initial_data_relation\": \""
            << (high->eikonal_mapped_initial_data
                ? "same physical events from solved ingoing eikonal and "
                  "unique marker-worldline crossings"
                : "none") << "\",\n"
            << "    \"coordinate_map_manifest_sha256\": "
            << (high->coordinate_map_manifest_sha256.empty()
                ? "null"
                : "\"" + high->coordinate_map_manifest_sha256 + "\"")
            << ",\n"
            << "    \"eikonal_boundary_radius\": "
            << high->eikonal_boundary_radius << ",\n"
            << "    \"eikonal_evolution_outer_radius\": "
            << high->eikonal_evolution_outer_radius << ",\n"
            << "    \"eikonal_analytic_vacuum_extension\": "
            << (high->eikonal_analytic_vacuum_extension
                ? "true" : "false") << ",\n"
            << "    \"eikonal_v_offset\": "
            << high->eikonal_v_offset << ",\n"
            << "    \"horizon_diagnostic\": \""
            << (options.coordinates == "eddington_finkelstein"
                ? "outermost_a_zero"
                : "polar_areal_gap_stop_proxy") << "\",\n"
            << "    \"inner_boundary\": \""
            << options.high_inner_boundary << "\",\n"
            << "    \"radial_grid\": \""
            << options.high_radial_grid << "\",\n"
            << "    \"markers\": " << options.high_markers << ",\n"
            << "    \"radial_cells\": " << options.high_radial_cells << ",\n"
            << "    \"r_min\": "
            << (options.high_inner_boundary == "origin_parity"
                    ? 0.0
                    : options.high_r_min)
            << ",\n"
            << "    \"r_max\": " << options.high_r_max << ",\n"
            << "    \"ef_excision_enabled\": "
            << (options.ef_enable_excision ? "true" : "false") << ",\n"
            << "    \"ef_require_vacuum_outer_boundary\": true,\n"
            << "    \"ef_evolution_status\": {\n"
            << "      \"available\": "
            << (options.coordinates == "eddington_finkelstein" ? "true" : "false") << ",\n"
            << "      \"requested_end_time_reached\": "
            << (high->ef_requested_end_time_reached ? "true" : "false") << ",\n"
            << "      \"final_advanced_time\": " << high->ef_final_advanced_time << ",\n"
            << "      \"termination_reason\": \""
            << json_escape(high->ef_termination_reason) << "\",\n"
            << "      \"termination_detail\": \""
            << json_escape(high->ef_termination_detail) << "\",\n"
            << "      \"last_accepted_dt\": " << high->ef_last_accepted_dt << ",\n"
            << "      \"final_trial_dt\": " << high->ef_final_trial_dt << ",\n"
            << "      \"minimum_active_radius\": " << high->ef_final_active_min_radius << ",\n"
            << "      \"first_field_center_radius\": " << high->ef_first_field_center_radius << ",\n"
            << "      \"first_cell_outer_radius\": " << high->ef_first_cell_outer_radius << ",\n"
            << "      \"active_markers_below_first_center\": "
            << high->ef_active_markers_below_first_field_center << ",\n"
            << "      \"active_weight_below_first_center\": "
            << high->ef_active_weight_inside_first_center << ",\n"
            << "      \"active_weight_inside_first_cell\": "
            << high->ef_active_weight_inside_first_cell << ",\n"
            << "      \"inner_face_mass\": " << high->ef_inner_face_mass << ",\n"
            << "      \"inner_face_mb\": " << high->ef_inner_face_mb << ",\n"
            << "      \"inner_face_b\": " << high->ef_inner_face_b << ",\n"
            << "      \"spatial_resolution_assessment\": \""
            << json_escape(high->ef_spatial_resolution_assessment) << "\",\n"
            << "      \"origin_mass_residual_available\": "
            << (high->ef_origin_mass_residual_available ? "true" : "false") << ",\n"
            << "      \"origin_mass_residual_over_adm\": "
            << high->ef_origin_mass_residual_over_adm << ",\n"
            << "      \"minimum_active_radius_over_first_cell_width\": "
            << high->ef_minimum_active_radius_over_first_cell_width << ",\n"
            << "      \"density_peak_radius\": "
            << high->ef_density_peak_radius << ",\n"
            << "      \"density_peak_half_maximum_cells\": "
            << high->ef_density_peak_half_maximum_cells << "\n"
            << "    },\n"
            << "    \"end_time\": " << options.high_t_end << ",\n"
            << "    \"dt_max\": " << options.high_dt_max << ",\n"
            << "    \"output_steps\": "
            << options.high_output_steps << ",\n"
            << "    \"gap_stop\": " << options.high_gap_stop << ",\n"
            << "    \"total_marker_mass\": "
            << options.high_total_mass << ",\n"
            << "    \"target_initial_adm_mass\": "
            << options.high_target_initial_adm_mass << ",\n"
            << "    \"fixed_family_adm_mass\": "
            << high->fixed_family_adm_mass << ",\n"
            << "    \"fixed_family_rest_weight\": "
            << high->fixed_family_rest_weight << ",\n"
            << "    \"fixed_family_binding_energy\": "
            << high->fixed_family_binding_energy << ",\n"
            << "    \"fixed_family_binding_fraction\": "
            << high->fixed_family_binding_fraction << ",\n"
            << "    \"fixed_family_binding_diagnostics_available\": "
            << (high->fixed_family_binding_diagnostics_available
                ? "true" : "false") << ",\n"
            << "    \"fixed_family_positive_binding_energy\": "
            << (high->fixed_family_positive_binding_energy
                ? "true" : "false")
            << ",\n"
            << "    \"initial_energy_at_infinity_proxy_definition\": "
               "\"E_proxy=exp(mu(t=0,r))*sqrt(1+w^2+2*eta/r^2); "
               "instantaneous frozen polar-areal metric; not conserved "
               "for dynamical data\",\n"
            << "    \"initial_energy_at_infinity_proxy_minimum\": "
            << high->initial_energy_at_infinity_proxy_minimum << ",\n"
            << "    \"initial_energy_at_infinity_proxy_maximum\": "
            << high->initial_energy_at_infinity_proxy_maximum << ",\n"
            << "    \"initial_energy_at_infinity_proxy_rest_weighted_mean\": "
            << high->initial_energy_at_infinity_proxy_rest_weighted_mean
            << ",\n"
            << "    \"initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one\": "
            << high->initial_energy_at_infinity_proxy_rest_weight_fraction_ge_one
            << ",\n"
            << "    \"initial_rest_weight_scale\": "
            << high->initial_rest_weight_scale << ",\n"
            << "    \"realized_initial_rest_weight\": "
            << high->initial_rest_weight << ",\n"
            << "    \"scale_radius\": "
            << options.high_scale_radius << ",\n"
            << "    \"initial_velocity_factor\": "
            << options.high_velocity_factor << ",\n"
            << "    \"initial_radial_infall\": "
            << options.high_radial_infall << ",\n"
            << "    \"infall_core_radius\": "
            << options.high_infall_core_radius << ",\n"
            << "    \"radial_importance_exponent\": "
            << options.high_radial_importance_exponent << ",\n"
            << "    \"shell_center\": "
            << options.high_shell_center << ",\n"
            << "    \"shell_width\": "
            << options.high_shell_width << ",\n"
            << "    \"shell_radial_importance_factor\": "
            << options.high_shell_radial_importance_factor << ",\n"
            << "    \"shell_radial_momentum\": "
            << options.high_shell_radial_momentum << ",\n"
            << "    \"shell_radial_momentum_dispersion\": "
            << options.high_shell_radial_momentum_dispersion << ",\n"
            << "    \"shell_tangential_velocity_dispersion\": "
            << options.high_shell_tangential_velocity_dispersion << ",\n"
            << "    \"fixed_specific_angular_momentum\": "
            << options.high_specific_angular_momentum << ",\n"
            << "    \"angular_momentum_scale\": "
            << options.high_angular_momentum_scale << ",\n"
            << "    \"sampling_realization\": "
            << options.high_sampling_realization << ",\n"
            << "    \"energy_cutoff_shell\": ";
        if (!high->energy_cutoff_continuum_available) {
            output << "null,\n";
        } else {
            output
                << "{\n"
                << "      \"ansatz\": \"f=C*(1-E/E0)_+^k*delta(ell-ell0)\",\n"
                << "      \"configured_amplitude_C\": "
                << high->energy_cutoff_configured_amplitude << ",\n"
                << "      \"configured_ell0\": "
                << high->energy_cutoff_configured_ell0 << ",\n"
                << "      \"configured_central_y\": "
                << high->energy_cutoff_configured_central_y << ",\n"
                << "      \"configured_cutoff_power_k\": "
                << high->energy_cutoff_configured_power << ",\n"
                << "      \"unscaled_adm_mass\": "
                << high->energy_cutoff_unscaled_adm_mass << ",\n"
                << "      \"continuum_rescaled_to_target_adm_mass\": "
                << (options.high_target_initial_adm_mass > 0.0
                    ? "true" : "false") << ",\n"
                << "      \"realized_amplitude_C\": "
                << high->energy_cutoff_realized_amplitude << ",\n"
                << "      \"realized_ell0\": "
                << high->energy_cutoff_realized_ell0 << ",\n"
                << "      \"realized_cutoff_power_k\": "
                << high->energy_cutoff_realized_power << ",\n"
                << "      \"cutoff_energy_E0\": "
                << high->energy_cutoff_cutoff_energy << ",\n"
                << "      \"inner_matter_radius\": "
                << high->energy_cutoff_inner_matter_radius << ",\n"
                << "      \"outer_matter_radius\": "
                << high->energy_cutoff_outer_matter_radius << ",\n"
                << "      \"continuum_adm_mass\": "
                << high->energy_cutoff_continuum_adm_mass << ",\n"
                << "      \"continuum_rest_weight\": "
                << high->energy_cutoff_continuum_rest_weight << ",\n"
                << "      \"continuum_binding_energy\": "
                << high->energy_cutoff_continuum_binding_energy << ",\n"
                << "      \"continuum_binding_fraction\": "
                << high->energy_cutoff_continuum_binding_fraction << ",\n"
                << "      \"central_lapse\": "
                << high->energy_cutoff_central_lapse << ",\n"
                << "      \"maximum_compactness\": "
                << high->energy_cutoff_maximum_compactness << ",\n"
                << "      \"minimum_compactness_gap\": "
                << high->energy_cutoff_minimum_compactness_gap << ",\n"
                << "      \"constraint_residual_linf\": "
                << high->energy_cutoff_constraint_residual_linf << ",\n"
                << "      \"anisotropic_jeans_residual_relative_l2\": "
                << high->energy_cutoff_jeans_residual_relative_l2 << ",\n"
                << "      \"sampling_rule\": \"shifted_stratified\",\n"
                << "      \"sampling_pair_count\": "
                << high->energy_cutoff_sampling_pair_count << ",\n"
                << "      \"sampling_count_semantics\": "
                   "\"exactly two adjacent signed-w markers per rank-one "
                   "lattice point; tensor GL orders are not used\",\n"
                << "      \"sampled_rest_weight\": "
                << high->energy_cutoff_sampled_rest_weight << ",\n"
                << "      \"sampled_rest_weight_relative_error\": "
                << high->energy_cutoff_sampled_rest_weight_relative_error
                << ",\n"
                << "      \"sampled_maximum_energy_over_cutoff\": "
                << high->energy_cutoff_sampled_maximum_energy_over_cutoff
                << ",\n"
                << "      \"marker_weight_measure\": "
                   "\"invariant_schwarzschild_wpic_rest_weight\",\n"
                << "      \"post_sampling_canonicalization_passes\": 0,\n"
                << "      \"zero_kick_pairing\": "
                   "\"adjacent (+w,-w) equal-weight pairs\",\n"
                << "      \"configured_inward_pair_mean_kick\": "
                << options.high_radial_infall << ",\n"
                << "      \"configured_kick_core_radius\": "
                << options.high_infall_core_radius << "\n"
                << "    },\n";
        }
        output
            << "    \"base_ic_fingerprint\": \"0x"
            << std::hex << std::setw(16) << std::setfill('0')
            << high->base_ic_fingerprint << std::dec << std::setfill(' ')
            << "\",\n"
            << "    \"realized_ic_fingerprint\": \"0x"
            << std::hex << std::setw(16) << std::setfill('0')
            << high->realized_ic_fingerprint << std::dec << std::setfill(' ')
            << "\",\n"
            << "    \"mass_weighted_mean_specific_angular_momentum\": "
            << high->mass_weighted_mean_specific_angular_momentum << ",\n"
            << "    \"mass_weighted_rms_specific_angular_momentum\": "
            << high->mass_weighted_rms_specific_angular_momentum << ",\n"
            << "    \"maximum_specific_angular_momentum\": "
            << high->maximum_specific_angular_momentum << ",\n"
            << "    \"timestep_omitted_mass_fraction\": "
            << options.timestep_omitted_mass_fraction << ",\n"
            << "    \"initial_min_gap\": "
            << high->initial_min_gap << ",\n"
            << "    \"final_min_gap\": "
            << high->final_min_gap << ",\n"
            << "    \"initial_min_lapse\": "
            << high->initial_min_lapse << ",\n"
            << "    \"final_min_lapse\": "
            << high->final_min_lapse << ",\n"
            << "    \"final_maximum_log_redshift\": "
            << high->final_maximum_log_redshift << ",\n"
            << "    \"initial_min_b\": "
            << high->initial_min_b << ",\n"
            << "    \"final_min_b\": "
            << high->final_min_b << ",\n"
            << "    \"initial_schwarzschild_map_a_linf\": "
            << high->initial_schwarzschild_map_a_linf << ",\n"
            << "    \"initial_schwarzschild_map_b_linf\": "
            << high->initial_schwarzschild_map_b_linf << ",\n"
            << "    \"initial_schwarzschild_map_m_linf\": "
            << high->initial_schwarzschild_map_m_linf << ",\n"
            << "    \"initial_schwarzschild_map_T_rr_relative_l2\": "
            << high->initial_schwarzschild_map_t_rr_relative_l2 << ",\n"
            << "    \"initial_schwarzschild_map_S_relative_l2\": "
            << high->initial_schwarzschild_map_s_relative_l2 << ",\n"
            << "    \"initial_schwarzschild_map_T_vv_relative_l2\": "
            << high->initial_schwarzschild_map_t_vv_relative_l2 << ",\n"
            << "    \"initial_schwarzschild_map_T_vr_relative_l2\": "
            << high->initial_schwarzschild_map_t_vr_relative_l2 << ",\n"
            << "    \"initial_schwarzschild_map_T_theta_theta_relative_l2\": "
            << high->initial_schwarzschild_map_t_theta_theta_relative_l2
            << ",\n"
            << "    \"initial_max_compactness\": "
            << high->initial_compactness << ",\n"
            << "    \"final_max_compactness\": "
            << high->final_compactness << ",\n"
            << "    \"initial_adm_mass\": "
            << high->initial_adm_mass << ",\n"
            << "    \"initial_deposited_adm_mass\": "
            << high->initial_deposited_adm_mass << ",\n"
            << "    \"null_slice_conserved_adm_mass\": "
            << high->null_slice_conserved_adm_mass << ",\n"
            << "    \"fixed_family_adm_relative_error\": "
            << (high->fixed_family_adm_mass > 0.0
                    && options.high_target_initial_adm_mass > 0.0
                ? std::abs(high->fixed_family_adm_mass
                        - options.high_target_initial_adm_mass)
                    / options.high_target_initial_adm_mass
                : -1.0) << ",\n"
            << "    \"null_slice_adm_relative_drift_from_family\": "
            << (high->fixed_family_adm_mass > 0.0
                    && high->null_slice_conserved_adm_mass > 0.0
                ? std::abs(high->null_slice_conserved_adm_mass
                        - high->fixed_family_adm_mass)
                    / high->fixed_family_adm_mass
                : -1.0) << ",\n"
            << "    \"initial_deposition_adm_relative_error_from_null_slice\": "
            << (high->null_slice_conserved_adm_mass > 0.0
                    && high->initial_deposited_adm_mass > 0.0
                ? std::abs(high->initial_deposited_adm_mass
                        - high->null_slice_conserved_adm_mass)
                    / high->null_slice_conserved_adm_mass
                : -1.0) << ",\n"
            << "    \"final_adm_mass\": "
            << high->final_adm_mass << ",\n"
            << "    \"adm_mass_relative_drift\": "
            << high->adm_mass_relative_drift << ",\n"
            << "    \"rest_mass_relative_error\": "
            << high->rest_mass_relative_error << ",\n"
            << "    \"excised_rest_mass\": "
            << high->excised_rest_mass << ",\n"
            << "    \"escaped_rest_mass\": "
            << high->escaped_rest_mass << ",\n"
            << "    \"apparent_horizon_radius\": "
            << high->apparent_horizon_radius << ",\n"
            << "    \"first_trapped_time\": "
            << high->first_trapped_time << ",\n"
            << "    \"post_horizon_advanced_time\": "
            << high->post_horizon_advanced_time << ",\n"
            << "    \"ef_post_horizon_evidence\": {\n"
            << "      \"schema\": \"venus_wpic_ef_post_horizon_evidence_v1\",\n"
            << "      \"available\": "
            << (options.coordinates == "eddington_finkelstein"
                ? "true" : "false") << ",\n"
            << "      \"accepted_step_audit_complete\": "
            << (high->ef_post_horizon_audit_complete
                ? "true" : "false") << ",\n"
            << "      \"accepted_step_audit_samples\": "
            << high->ef_accepted_step_audit_samples << ",\n"
            << "      \"minimum_p_r_over_evolution\": "
            << high->ef_minimum_p_r_over_evolution << ",\n"
            << "      \"minimum_b_over_evolution\": "
            << high->ef_minimum_b_over_evolution << ",\n"
            << "      \"finite_fields_throughout\": "
            << (high->ef_finite_fields_throughout
                ? "true" : "false") << ",\n"
            << "      \"fixed_adm_bookkeeping_throughout\": "
            << (high->ef_fixed_adm_bookkeeping_throughout
                ? "true" : "false") << ",\n"
            << "      \"causal_excision_valid_throughout\": "
            << (high->ef_causal_excision_valid_throughout
                ? "true" : "false") << ",\n"
            << "      \"configured_excision_a_threshold\": "
            << options.ef_excision_a << ",\n"
            << "      \"configured_excision_buffer_cells\": "
            << options.ef_excision_buffer_cells << ",\n"
            << "      \"configured_excision_horizon_fraction\": "
            << options.ef_excision_horizon_fraction << ",\n"
            << "      \"rho_infall_observer\": \"u^v=1/b; u^r=(a-1)/2; unit timelike, generally not geodesic\",\n"
            << "      \"rho_infall_deposition\": \"sum dm*(p_r+(1+2*eta/r^2)/p_r)^2/(4*p_r*dV); active markers only\",\n"
            << "      \"maximum_excision_radius\": "
            << high->ef_maximum_excision_radius << ",\n"
            << "      \"maximum_excised_rest_weight\": "
            << high->ef_maximum_excised_rest_weight << ",\n"
            << "      \"causal_excision_inside_horizon_seen\": "
            << (high->ef_causal_excision_inside_horizon_seen
                ? "true" : "false") << ",\n"
            << "      \"excised_interior_rest_weight_seen\": "
            << (high->ef_excised_interior_rest_weight_seen
                ? "true" : "false") << "\n"
            << "    },\n"
            << "    \"initial_r50\": " << high->initial_r50 << ",\n"
            << "    \"minimum_r50\": " << high->minimum_r50 << ",\n"
            << "    \"final_r50\": " << high->final_r50 << ",\n"
            << "    \"final_virial_ratio\": "
            << high->final_virial_ratio << ",\n"
            << "    \"virial_ratio_interpretation\": "
               "\"Newtonian 2K/abs(W) proxy; not a relativistic virial "
               "identity\",\n"
            << "    \"final_anisotropic_jeans_relative_l1\": "
            << high->final_anisotropic_jeans_relative_l1 << ",\n"
            << "    \"anisotropic_jeans_residual_definition\": "
               "\"weighted_L1[p_r_prime+(rho+p_r)mu_r-"
               "(p_t_sum-2p_r)/r]/weighted_L1[absolute_term_sum]; "
               "vacuum and derivative-boundary cells excluded\",\n"
            << "    \"halo_classifier\": ";
        write_halo_classification_json(
            output, high->halo_classification);
        output
            << ",\n    \"horizon_time\": " << high->horizon_time << ",\n"
            << "    \"steps\": " << high->steps << ",\n"
            << "    \"horizon_formed\": "
            << (high->horizon_formed ? "true" : "false")
            << ",\n    \"evolved_past_horizon\": "
            << (high->evolved_past_horizon ? "true" : "false")
            << ",\n    \"virialized_halo\": "
            << (high->virialized_halo ? "true" : "false")
            << ",\n    \"outcome\": \"" << high->outcome << "\""
            << "\n  }";
    }
    output << "\n}\n";
}

}  // namespace

// ---- Scenario dispatch ----------------------------------------------------
int ev::apps::run_weighted_einstein_vlasov_pic(int argc, char** argv) {
    try {
        const RunOptions options = parse_options(argc, argv);
        std::filesystem::create_directories(options.output_dir);
        LowSummary low;
        HighSummary high;
        const bool run_low =
            options.scenario == "all" || options.scenario == "low";
        const bool run_high =
            options.scenario == "all" || options.scenario == "high";
        if (run_low) {
            low = run_low_compactness(options);
        }
        if (run_high) {
            high = options.coordinates == "eddington_finkelstein"
                ? run_high_compactness_eddington_finkelstein(options)
                : run_high_compactness_schwarzschild(options);
        }
        write_summary(
            options, run_low ? &low : nullptr, run_high ? &high : nullptr);

        // Compare the measured outcome with the requested experiment only
        // after evolution. An expectation never changes the field or forces.
        bool passed = true;
        if (run_low) {
            const bool require_newtonian_agreement =
                low.initial_compactness < 0.01;
            std::cout
                << "Low compactness: virialized_halo="
                << std::boolalpha << low.virialized_halo
                << " agrees_with_vlasov_poisson="
                << low.agrees_with_vlasov_poisson
                << " require_newtonian_agreement="
                << require_newtonian_agreement
                << " density_L1=" << low.density_l1_difference
                << " final_Q=" << low.final_virial_ratio << '\n';
            passed = passed
                && low.virialized_halo
                && (!require_newtonian_agreement
                    || low.agrees_with_vlasov_poisson)
                && low.ev_rest_mass_relative_error < 1.0e-12
                && low.vp_rest_mass_relative_error < 1.0e-12;
        }
        if (run_high) {
            std::cout
                << "High compactness: coordinates=" << options.coordinates
                << " outcome=" << high.outcome
                << " horizon_formed="
                << std::boolalpha << high.horizon_formed
                << " horizon_time=" << high.horizon_time
                << " final_gap=" << high.final_min_gap
                << " final_Cmax=" << high.final_compactness << '\n';
            const bool expected_outcome_observed =
                options.initialize_only
                || options.expected_outcome == "any"
                || (options.expected_outcome == "black_hole"
                    && high.horizon_formed
                    && high.outcome != "interior_evolution_limit"
                    && high.outcome != "ef_outer_boundary_evolution_limit"
                    && high.outcome != "sc_outer_boundary_evolution_limit")
                || (options.expected_outcome == "halo"
                    && high.virialized_halo);
            passed = passed
                && expected_outcome_observed
                && high.adm_mass_relative_drift < 0.01
                && high.rest_mass_relative_error < 1.0e-12;
        }
        std::cout << "Einstein--Vlasov WPIC output: "
                  << std::filesystem::absolute(options.output_dir) << '\n';
        return passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        return 1;
    }
}
