// Evolve a spherical particle distribution with Newtonian gravity in a static
// or prescribed expanding background. Fresh data come from the power-law
// sampler below. A native restart instead supplies the actual particle state.
// Each marker stores physical radius r, peculiar radial velocity w, angular
// momentum coordinate eta=L^2/2, mass, identity and subhalo internal energy.
// Physical radial velocity is H*r+w. The subhalo model treats marker masses
// as the masses of the interacting objects, so unequal weights enter its
// encounter laws directly.
//
// Read this driver in the order of a simulation step. Deposit particle mass
// and velocity moments on radial shells, solve Poisson's equation, choose the
// next orbit/event interval, and advance the trajectories with KDK. At an
// encounter deadline, sample subhalo scattering or capture at the current
// radii. Capture transfers relative orbital energy into the remnant's internal
// energy. Deposit the changed particles and solve gravity again before the
// next orbit step. The histories record both the physical evolution and the
// numerical accuracy conditions. Checkpoints preserve particles, event ages
// and the random state needed to resume the same calculation.
//
// A zero collision strength follows the collisionless branch. This CPU
// driver and the CUDA driver each select their own orbit and event intervals.
// Their settings must be compared explicitly when comparing trajectories.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "ev/canonical_open_carrier_compression.hpp"
#include "ev/restart_output_guard.hpp"
#include "ev/sha256.hpp"
#include "ev/weighted_vlasov_pic.hpp"
#include "ev/weighted_vlasov_pic_event_scheduler.hpp"
#include "ev/weighted_vlasov_pic_population_control.hpp"
#include "ev/weighted_vlasov_pic_subhalo_collisions.hpp"
#include "ev/wpic_task_profile.hpp"
#include "ev/apps/wpic_driver_entry.hpp"

namespace {

std::atomic<bool> stop_requested{false};

void request_stop(int) {
    stop_requested.store(true, std::memory_order_relaxed);
}

struct Options {
    double gamma = 1.5;
    // A descriptive native-input label; the checkpoint owns the particles.
    // The power-law sampler and its existing restart contract stay the default.
    std::string initial_condition_label = "powerlaw";
    bool boundary_diagnostics = false;
    // Sachiet's background controls retain the original Einstein--de Sitter
    // defaults. The static choice sets alpha=0 before constructing any IC,
    // field, or orbit coefficient, so these all use the same background.
    std::string vp_universe = "expanding";
    double vp_alpha = 2.0 / 3.0;
    double vp_time_offset = 12.0;
    // Scale the physical initial velocity u=H*r+w, including tangential
    // motion, to prepare controlled departures from orbital support. This
    // acts on a fresh sample once; resumed marker velocities are preserved.
    double initial_velocity_factor = 1.0;
    double end_time = 2.0;
    double maximum_timestep = 4.0;
    double dynamical_fraction = 0.04;
    double output_interval = 1.0;
    int history_every = 100;
    int restart_every = 1000;
    int particles_per_radial_cell = 2048;
    // The logarithmic field/collision mesh sets the physical shell width
    // Delta r used by the locality diagnostic.  Increasing Nr, or focusing
    // [r_min,r_max] on the dynamically relevant halo, can resolve a shorter
    // mean free path without changing the collision model or its threshold.
    int radial_cells = 256;
    double radial_minimum = 0.1;
    double radial_maximum = 1000.0;
    double low_angular_tail_probability = 0.0;
    double low_angular_tail_fraction = 0.0;
    double low_angular_source_rmax = 0.0;
    int low_angular_source_ppc_multiplier = 1;
    double subhalo_collision_strength = 0.0;
    double subhalo_velocity_scale = 0.1;
    double subhalo_plummer_area_per_mass = 1.0;
    double subhalo_low_speed_regularization = 0.05;
    // This is a hard ceiling, not necessarily the applied interval.  With a
    // positive timescale fraction, the scheduler shortens it in response to
    // the local collision, dynamical, or potential-change clock.
    double subhalo_collision_interval = 5.0;
    double subhalo_collision_timescale_fraction = 0.1;
    double subhalo_collision_cfl = 0.2;
    int subhalo_collision_maximum_substeps = 256;
    // Give every disjoint physical collision cell its own CFL clock inside
    // the globally scheduled Boltzmann operator batch.
    bool subhalo_adaptive_cell_timestep = false;
    bool collision_prestep_coordinate_due = false;
    // Zero keeps locality diagnostic-only.  A positive value checkpoints and
    // exits with status 98 as soon as any physical radial collision cell
    // exceeds the requested Delta-r/lambda_mfp ratio.
    double subhalo_locality_maximum_ratio = 0.0;
    int subhalo_inner_collision_cells = 1;
    int subhalo_collision_lanes = 1;
    std::string subhalo_pair_sampling = "maximum_weight_importance";
    std::uint64_t subhalo_collision_seed =
        UINT64_C(0x73756268616c6f31);
    // Exact-clone refinement controls are read for parameter and restart
    // validation. A positive trigger is rejected: coincident daughters create
    // additional same-parent collision pairs and change the physical collision
    // rate. The supported trigger is zero.
    std::size_t subhalo_repopulation_trigger_live_markers = 0;
    int subhalo_repopulation_refinement_factor = 2;
    std::size_t subhalo_compression_trigger_markers = 0;
    std::size_t subhalo_compression_minimum_markers = 0;
    int subhalo_compression_phase_subcells = 4;
    bool subhalo_compression_preserve_source_cell = false;
    double subhalo_compression_minimum_reduction_fraction = 0.1;
    double subhalo_compression_tolerance = 2.0e-10;
    bool subhalo_compression_minimize_weight_variance = true;
    double subhalo_compression_maximum_mass_amplification = 1.0;
    unsigned int threads = 1;
    std::filesystem::path output_directory = "weighted_pic_powerlaw";
    std::filesystem::path restart;
    bool show_help = false;
};

// ---- Runtime options and physical parameter translation --------------------
void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "\n"
#if defined(VENUS_CPU_EXPANDING_WPIC_VP_PROFILE)
        << "Locked task profile: collisionless expanding Vlasov-Poisson "
           "WPIC.\n"
        << "--subhalo-collision-strength is pinned to 0; every nonzero value "
           "is rejected.\n\n"
#elif defined(VENUS_CPU_EXPANDING_WPIC_SUBHALO_PROFILE)
        << "Locked task profile: expanding collisional subhalo WPIC.\n"
        << "--subhalo-collision-strength must be strictly positive; zero is "
           "rejected.\n\n"
#else
        << "Compatibility driver: strength 0 selects collisionless VP and a "
           "positive strength selects subhalo collisions.\n\n"
#endif
        << "Radial mesh controls for fresh resolution studies:\n"
        << "  --radial-cells N       logarithmic radial cells (default: 256)\n"
        << "  --radial-minimum R     first resolved radius (default: 0.1)\n"
        << "  --radial-maximum R     outer resolved radius (default: 1000)\n"
        << "  --initial-condition-label NAME  describe an explicit native input [powerlaw]\n"
        << "  --boundary-diagnostics  count accepted reflecting impacts and radius maxima [off]\n"
        << "\n"
        << "A finer mesh reduces the physical collision-cell width Delta r; "
           "it does not relax\n"
        << "the Delta-r/lambda_mfp locality test.  When continuing a restart, "
           "repeat the\n"
        << "same radial mesh values used by the fresh run.\n"
        << "\n"
        << "Background controls (fresh runs and matching restarts):\n"
        << "  --vp-universe expanding|static     default: expanding\n"
        << "  --vp-alpha X                       0 <= X <= 1; default: 2/3\n"
        << "  --vp-time-offset X                 positive; default: 12\n"
        << "  Static mode sets alpha=0; the supplied density is a cusp IC.\n"
        << "  --initial-velocity-factor F       positive physical IC velocity factor; default: 1\n"
        << "  Fresh data use u_new=F*u and eta_new=F^2*eta. Repeat F on resume.\n"
        << "\n"
        << "Collision-clock controls:\n"
        << "  --subhalo-collision-interval X       global operator-batch cap\n"
        << "  --subhalo-collision-timescale-fraction X\n"
        << "  --subhalo-collision-cfl X\n"
        << "  --subhalo-collision-maximum-substeps N\n"
        << "  --subhalo-adaptive-cell-timestep     independent cell CFL clocks\n"
        << "  --collision-prestep-coordinate-due   apply safe coordinate-close deadlines before KDK\n"
        << "\n"
        << "Use --help or -h to print this summary.  Established orbit, "
           "marker, output, restart, and task-compatible collision controls "
           "remain available.\n";
}

Options parse_options(int argc, char** argv) {
    Options result;
    auto value = [&](int& argument, const char* name) {
        if (++argument >= argc) {
            throw std::runtime_error(std::string("Missing value for ") + name);
        }
        return std::string(argv[argument]);
    };
    for (int argument = 1; argument < argc; ++argument) {
        const std::string name = argv[argument];
        if (name == "--gamma") {
            result.gamma = std::stod(value(argument, "--gamma"));
        } else if (name == "--initial-condition-label") {
            result.initial_condition_label = value(argument, "--initial-condition-label");
        } else if (name == "--boundary-diagnostics") {
            result.boundary_diagnostics = true;
        } else if (name == "--vp-universe") {
            result.vp_universe = value(argument, "--vp-universe");
        } else if (name == "--vp-alpha") {
            result.vp_alpha = std::stod(value(argument, "--vp-alpha"));
        } else if (name == "--vp-time-offset") {
            result.vp_time_offset = std::stod(value(argument, "--vp-time-offset"));
        } else if (name == "--initial-velocity-factor") {
            result.initial_velocity_factor =
                std::stod(value(argument, "--initial-velocity-factor"));
        } else if (name == "--end-time") {
            result.end_time = std::stod(value(argument, "--end-time"));
        } else if (name == "--maximum-timestep") {
            result.maximum_timestep =
                std::stod(value(argument, "--maximum-timestep"));
        } else if (name == "--dynamical-fraction") {
            result.dynamical_fraction =
                std::stod(value(argument, "--dynamical-fraction"));
        } else if (name == "--output-interval") {
            result.output_interval =
                std::stod(value(argument, "--output-interval"));
        } else if (name == "--history-every") {
            result.history_every = std::stoi(value(argument, "--history-every"));
        } else if (name == "--restart-every") {
            result.restart_every = std::stoi(value(argument, "--restart-every"));
        } else if (name == "--particles-per-radial-cell") {
            result.particles_per_radial_cell = std::stoi(
                value(argument, "--particles-per-radial-cell"));
        } else if (name == "--radial-cells") {
            result.radial_cells =
                std::stoi(value(argument, "--radial-cells"));
        } else if (name == "--radial-minimum") {
            result.radial_minimum =
                std::stod(value(argument, "--radial-minimum"));
        } else if (name == "--radial-maximum") {
            result.radial_maximum =
                std::stod(value(argument, "--radial-maximum"));
        } else if (name == "--low-angular-tail-probability") {
            result.low_angular_tail_probability = std::stod(
                value(argument, "--low-angular-tail-probability"));
        } else if (name == "--low-angular-tail-fraction") {
            result.low_angular_tail_fraction = std::stod(
                value(argument, "--low-angular-tail-fraction"));
        } else if (name == "--low-angular-source-rmax") {
            result.low_angular_source_rmax = std::stod(
                value(argument, "--low-angular-source-rmax"));
        } else if (name == "--low-angular-source-ppc-multiplier") {
            result.low_angular_source_ppc_multiplier = std::stoi(
                value(argument, "--low-angular-source-ppc-multiplier"));
        } else if (name == "--subhalo-collision-strength") {
            result.subhalo_collision_strength = std::stod(
                value(argument, "--subhalo-collision-strength"));
        } else if (name == "--subhalo-velocity-scale") {
            result.subhalo_velocity_scale = std::stod(
                value(argument, "--subhalo-velocity-scale"));
        } else if (name == "--subhalo-plummer-area-per-mass") {
            result.subhalo_plummer_area_per_mass = std::stod(
                value(argument, "--subhalo-plummer-area-per-mass"));
        } else if (name == "--subhalo-low-speed-regularization") {
            result.subhalo_low_speed_regularization = std::stod(
                value(argument, "--subhalo-low-speed-regularization"));
        } else if (name == "--subhalo-collision-interval") {
            result.subhalo_collision_interval = std::stod(
                value(argument, "--subhalo-collision-interval"));
        } else if (name == "--subhalo-collision-timescale-fraction") {
            result.subhalo_collision_timescale_fraction = std::stod(
                value(
                    argument,
                    "--subhalo-collision-timescale-fraction"));
        } else if (name == "--subhalo-collision-cfl") {
            result.subhalo_collision_cfl = std::stod(
                value(argument, "--subhalo-collision-cfl"));
        } else if (name == "--subhalo-collision-maximum-substeps") {
            result.subhalo_collision_maximum_substeps = std::stoi(
                value(argument, "--subhalo-collision-maximum-substeps"));
        } else if (name == "--collision-prestep-coordinate-due") {
            result.collision_prestep_coordinate_due = true;
        } else if (name == "--subhalo-adaptive-cell-timestep") {
            result.subhalo_adaptive_cell_timestep = true;
        } else if (name == "--subhalo-locality-maximum-ratio") {
            result.subhalo_locality_maximum_ratio = std::stod(
                value(argument, "--subhalo-locality-maximum-ratio"));
        } else if (name == "--subhalo-inner-collision-cells") {
            result.subhalo_inner_collision_cells = std::stoi(
                value(argument, "--subhalo-inner-collision-cells"));
        } else if (name == "--subhalo-collision-lanes") {
            result.subhalo_collision_lanes = std::stoi(
                value(argument, "--subhalo-collision-lanes"));
        } else if (name == "--subhalo-pair-sampling") {
            result.subhalo_pair_sampling =
                value(argument, "--subhalo-pair-sampling");
        } else if (name == "--subhalo-collision-seed") {
            result.subhalo_collision_seed =
                static_cast<std::uint64_t>(std::stoull(
                    value(argument, "--subhalo-collision-seed"),
                    nullptr,
                    0));
        } else if (
            name == "--subhalo-repopulation-trigger-live-markers"
            || name == "--subhalo-reinjection-trigger-markers") {
            result.subhalo_repopulation_trigger_live_markers =
                static_cast<std::size_t>(std::stoull(
                    value(
                        argument,
                        "--subhalo-repopulation-trigger-live-markers")));
        } else if (
            name == "--subhalo-repopulation-refinement-factor"
            || name == "--subhalo-reinjection-factor") {
            result.subhalo_repopulation_refinement_factor = std::stoi(
                value(
                    argument,
                    "--subhalo-repopulation-refinement-factor"));
        } else if (name == "--subhalo-compression-trigger-markers") {
            result.subhalo_compression_trigger_markers =
                static_cast<std::size_t>(std::stoull(
                    value(argument, "--subhalo-compression-trigger-markers")));
        } else if (name == "--subhalo-compression-minimum-markers") {
            result.subhalo_compression_minimum_markers =
                static_cast<std::size_t>(std::stoull(
                    value(argument, "--subhalo-compression-minimum-markers")));
        } else if (name == "--subhalo-compression-phase-subcells") {
            result.subhalo_compression_phase_subcells = std::stoi(
                value(argument, "--subhalo-compression-phase-subcells"));
        } else if (name
                   == "--subhalo-compression-preserve-source-cell") {
            result.subhalo_compression_preserve_source_cell = true;
        } else if (name == "--subhalo-compression-minimum-reduction-fraction") {
            result.subhalo_compression_minimum_reduction_fraction = std::stod(
                value(
                    argument,
                    "--subhalo-compression-minimum-reduction-fraction"));
        } else if (name == "--subhalo-compression-tolerance") {
            result.subhalo_compression_tolerance = std::stod(
                value(argument, "--subhalo-compression-tolerance"));
        } else if (
            name == "--subhalo-compression-legacy-minimum-step") {
            result.subhalo_compression_minimize_weight_variance = false;
        } else if (
            name == "--subhalo-compression-maximum-mass-amplification") {
            result.subhalo_compression_maximum_mass_amplification =
                std::stod(value(
                    argument,
                    "--subhalo-compression-maximum-mass-amplification"));
        } else if (name == "--threads") {
            result.threads = static_cast<unsigned int>(
                std::stoul(value(argument, "--threads")));
        } else if (name == "--output-directory") {
            result.output_directory = value(argument, "--output-directory");
        } else if (name == "--restart") {
            result.restart = value(argument, "--restart");
        } else if (name == "--help" || name == "-h") {
            result.show_help = true;
        } else {
            throw std::runtime_error(
                "Unknown expanding WPIC option: " + name
                + " (use --help for radial-mesh controls)");
        }
    }
    if (result.show_help) {
        return result;
    }
    if ((result.vp_universe != "expanding" && result.vp_universe != "static")
        || !std::isfinite(result.vp_alpha)
        || !(result.vp_alpha >= 0.0 && result.vp_alpha <= 1.0)
        || !std::isfinite(result.vp_time_offset)
        || !(result.vp_time_offset > 0.0)) {
        throw std::runtime_error(
            "Background requires --vp-universe expanding|static, "
            "finite 0 <= --vp-alpha <= 1, and positive --vp-time-offset.");
    }
    if (result.vp_universe == "static") result.vp_alpha = 0.0;
    if (!(result.initial_velocity_factor > 0.0)
        || !std::isfinite(result.initial_velocity_factor)) {
        throw std::runtime_error(
            "--initial-velocity-factor must be finite and strictly positive.");
    }
#if defined(VENUS_CPU_EXPANDING_WPIC_VP_PROFILE)
    try {
        ev::validate_expanding_wpic_task_profile(
            ev::ExpandingWPICTaskProfile::collisionless_vlasov_poisson,
            result.subhalo_collision_strength);
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        std::exit(1);
    }
#elif defined(VENUS_CPU_EXPANDING_WPIC_SUBHALO_PROFILE)
    try {
        ev::validate_expanding_wpic_task_profile(
            ev::ExpandingWPICTaskProfile::collisional_subhalo,
            result.subhalo_collision_strength);
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        std::exit(1);
    }
#endif
    if (result.radial_cells < 16) {
        throw std::runtime_error("--radial-cells must be at least 16.");
    }
    if (!(result.radial_minimum > 0.0)
        || !std::isfinite(result.radial_minimum)) {
        throw std::runtime_error(
            "--radial-minimum must be finite and strictly positive.");
    }
    if (!(result.radial_maximum > result.radial_minimum)
        || !std::isfinite(result.radial_maximum)) {
        throw std::runtime_error(
            "--radial-maximum must be finite and exceed --radial-minimum.");
    }
    // gamma specifies the initial density index. The explicit supported
    // values select the available halo examples. Later density slopes follow
    // the evolution and are measured independently from the output.
    const bool supported_gamma = result.gamma == 0.1 || result.gamma == 0.3
        || result.gamma == 0.5 || result.gamma == 0.8
        || result.gamma == 1.2 || result.gamma == 1.5 || result.gamma == 2.0
        || result.gamma == 2.5;
    if (result.initial_condition_label.empty()
        || result.initial_condition_label.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
            != std::string::npos
        || (result.initial_condition_label != "powerlaw" && result.restart.empty())) {
        throw std::runtime_error(
            "A custom --initial-condition-label requires native --restart input "
            "and letters, digits, underscores or hyphens.");
    }
    if (!supported_gamma || !(result.end_time > 0.0)
        || !(result.maximum_timestep > 0.0)
        || !(result.dynamical_fraction > 0.0)
        || !(result.dynamical_fraction < 0.5)
        || !(result.output_interval > 0.0)
        || result.history_every < 1 || result.restart_every < 1
        || result.particles_per_radial_cell < 128
        || result.particles_per_radial_cell % 2 != 0
        || !(result.low_angular_source_rmax >= 0.0)
        || !std::isfinite(result.low_angular_source_rmax)
        || result.low_angular_source_ppc_multiplier < 1
        || result.low_angular_source_ppc_multiplier > 64
        || !(result.subhalo_collision_strength >= 0.0)
        || !std::isfinite(result.subhalo_collision_strength)
        || !(result.subhalo_velocity_scale > 0.0)
        || !std::isfinite(result.subhalo_velocity_scale)
        || !(result.subhalo_plummer_area_per_mass > 0.0)
        || !std::isfinite(result.subhalo_plummer_area_per_mass)
        || !(result.subhalo_low_speed_regularization
             >= ev::weighted_pic_subhalo_plummer_table::speed.front())
        || !(result.subhalo_low_speed_regularization < 1.0)
        || !std::isfinite(result.subhalo_low_speed_regularization)
        || !(result.subhalo_collision_interval > 0.0)
        || !std::isfinite(result.subhalo_collision_interval)
        || !(result.subhalo_collision_timescale_fraction >= 0.0)
        || !(result.subhalo_collision_timescale_fraction <= 1.0)
        || !std::isfinite(
            result.subhalo_collision_timescale_fraction)
        || !(result.subhalo_collision_cfl > 0.0)
        || !std::isfinite(result.subhalo_collision_cfl)
        || result.subhalo_collision_maximum_substeps < 1
        || !(result.subhalo_locality_maximum_ratio >= 0.0)
        || !std::isfinite(result.subhalo_locality_maximum_ratio)
        || result.subhalo_inner_collision_cells < 1
        || result.subhalo_inner_collision_cells > 256
        || result.subhalo_collision_lanes < 1
        || result.subhalo_collision_lanes > 256
        || (result.subhalo_pair_sampling != "maximum_weight_rejection"
            && result.subhalo_pair_sampling
                != "maximum_weight_importance")
        || result.subhalo_repopulation_refinement_factor < 2
        || result.subhalo_repopulation_refinement_factor > 64
        || result.subhalo_compression_phase_subcells < 1
        || result.subhalo_compression_phase_subcells > 64
        || !(result.subhalo_compression_minimum_reduction_fraction >= 0.0)
        || !(result.subhalo_compression_minimum_reduction_fraction < 1.0)
        || !std::isfinite(
            result.subhalo_compression_minimum_reduction_fraction)
        || !(result.subhalo_compression_tolerance > 0.0)
        || !std::isfinite(result.subhalo_compression_tolerance)
        || !(result.subhalo_compression_maximum_mass_amplification >= 1.0)
        || !std::isfinite(
            result.subhalo_compression_maximum_mass_amplification)
        || result.threads == 0) {
        throw std::runtime_error("Invalid expanding WPIC run options.");
    }
    const bool angular_importance_disabled =
        result.low_angular_tail_probability == 0.0
        && result.low_angular_tail_fraction == 0.0;
    const bool angular_importance_valid =
        result.low_angular_tail_probability > 0.0
        && result.low_angular_tail_probability < 1.0
        && result.low_angular_tail_fraction
            > result.low_angular_tail_probability
        && result.low_angular_tail_fraction < 1.0;
    if (!angular_importance_disabled && !angular_importance_valid) {
        throw std::runtime_error(
            "Low-angular-momentum importance sampling requires "
            "0 < tail probability < tail fraction < 1, or both values zero.");
    }
    if (angular_importance_disabled
        && (result.low_angular_source_rmax != 0.0
            || result.low_angular_source_ppc_multiplier != 1)) {
        throw std::runtime_error(
            "Low-angular source targeting requires angular importance sampling.");
    }
    if (result.subhalo_repopulation_trigger_live_markers > 0
        && result.subhalo_collision_strength == 0.0) {
        throw std::runtime_error(
            "Subhalo population control is meaningful only when capture "
            "collisions are enabled.");
    }
    if (result.subhalo_repopulation_trigger_live_markers > 0) {
        throw std::runtime_error(
            "Exact-clone subhalo repopulation is disabled: same-parent "
            "daughter pairs add a finite diagonal collision-rate term. "
            "Implement persistent lineage exclusion or a validated "
            "instantaneous-DF resampler before enabling this option.");
    }
    if (result.subhalo_repopulation_trigger_live_markers > 0
        && result.subhalo_compression_trigger_markers > 0) {
        throw std::runtime_error(
            "Subhalo population refinement and marker compression cannot be "
            "enabled in the same run.");
    }
    if (result.collision_prestep_coordinate_due) {
        if (!(result.subhalo_collision_strength > 0.0)
            || !(result.subhalo_collision_timescale_fraction > 0.0)) {
            throw std::runtime_error(
                "Pre-step coordinate due decisions require adaptive subhalo collisions.");
        }
        if (std::filesystem::exists(result.output_directory)
            && !std::filesystem::is_empty(result.output_directory)) {
            throw std::runtime_error(
                "Pre-step coordinate due decisions require a fresh output segment.");
        }
    }
    return result;
}

// Translate the command-line choices into the shared field and orbit settings.
// The radial mesh, background and boundaries must agree because the same
// particles supply the Poisson source and move through its reconstructed force.
ev::Params make_parameters(const Options& run) {
    ev::Params p;
    p.physics_model = "vlasov_poisson";
    p.vp_universe = run.vp_universe;
    p.vp_alpha = run.vp_alpha;
    p.vp_time_offset = run.vp_time_offset;
    p.vp_gravity = 1.0;
    p.vp_poisson_source = "density_contrast";
    // The PIC field bridge uses the conservative radial control volumes.  The
    // marker pusher below is independent of the semi-Lagrangian remap, but this
    // flag selects the same conservative geometry and field interpolation.
    p.vp_interpolation = "hamiltonian_sldg";
    p.vp_orbit_regularization = "off";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "uniform";
    p.Nr = run.radial_cells;
    p.Nw = 4;
    p.NL = 2;
    p.r_min = run.radial_minimum;
    p.r_max = run.radial_maximum;
    p.radial_grid = "log";
    p.w_max = 3.0;
    p.L_max = 8.0;
    p.w_stretch = 0.0;
    p.inner_boundary = "origin_parity";
    p.outer_boundary = "reflecting";
    p.integrator = "semi_lagrangian_kick_drift_kick";
    p.threads = static_cast<int>(run.threads);
    return p;
}

// Keep physical pair-law parameters (strength, area and velocity scale)
// separate from numerical controls (CFL, substeps, cells and pair sampling).
ev::WeightedPICSubhaloCollisionOptions make_collision_options(
    const Options& run) {
    ev::WeightedPICSubhaloCollisionOptions options;
    options.velocity_scale = run.subhalo_velocity_scale;
    options.plummer_area_per_mass =
        run.subhalo_plummer_area_per_mass;
    options.low_speed_regularization =
        run.subhalo_low_speed_regularization;
    options.strength = run.subhalo_collision_strength;
    options.collision_cfl = run.subhalo_collision_cfl;
    options.maximum_substeps =
        run.subhalo_collision_maximum_substeps;
    options.adaptive_cell_timestep =
        run.subhalo_adaptive_cell_timestep;
    options.inner_collision_cells =
        run.subhalo_inner_collision_cells;
    options.collision_lanes =
        run.subhalo_collision_lanes;
    options.pair_sampling =
        run.subhalo_pair_sampling == "maximum_weight_importance"
        ? ev::WeightedPICSubhaloPairSampling::maximum_weight_importance
        : ev::WeightedPICSubhaloPairSampling::maximum_weight_rejection;
    options.minimum_cell_population = 2;
    return options;
}

// ---- Self-similar power-law initial condition and quiet marker sampling ----
class PowerLawInitialCondition {
public:
    PowerLawInitialCondition(double gamma, const ev::WeightedPICRadialMesh& mesh)
        : gamma_(gamma), mesh_(mesh) {
        long double shape_mass = 0.0L;
        for (std::size_t cell = 0; cell < mesh_.shell_volume.size(); ++cell) {
            shape_mass += integrate_shape_mass(
                mesh_.edge[cell], mesh_.edge[cell + 1]);
        }
        // Normalize the resolved density to mass 0.8. Keep the unresolved
        // uniform interior separate so it can enter the inner Poisson source.
        density_scale_ = 0.8 / static_cast<double>(shape_mass);
        shell_mass_.resize(mesh_.shell_volume.size(), 0.0);
        enclosed_mass_at_left_.resize(mesh_.shell_volume.size(), 0.0);
        // Store each shell mass and the mass inside its left face. A sampled
        // particle then knows its enclosed mass without another integration.
        double enclosed = 0.0;
        for (std::size_t cell = 0; cell < shell_mass_.size(); ++cell) {
            enclosed_mass_at_left_[cell] = enclosed;
            shell_mass_[cell] = density_scale_ * static_cast<double>(
                integrate_shape_mass(mesh_.edge[cell], mesh_.edge[cell + 1]));
            enclosed += shell_mass_[cell];
        }
        outer_mass_ = enclosed;
        central_mass_ = 4.0 * ev::PI / 3.0
            * std::pow(mesh_.edge.front(), 3) * density(mesh_.edge.front());
    }

    double density(double radius) const {
        return density_scale_ * shape(radius);
    }

    double shell_mass(std::size_t cell) const {
        return shell_mass_[cell];
    }

    double enclosed_mass_at_left(std::size_t cell) const {
        return enclosed_mass_at_left_[cell];
    }

    double outer_mass() const {
        return outer_mass_;
    }

    double central_mass() const {
        return central_mass_;
    }

    // Tabulate the fraction of this shell mass inside each subdivision.
    // Inverting this cumulative mass fraction samples rho(r)*4*pi*r^2,
    // which differs from sampling radius uniformly.
    std::vector<double> radial_cdf(std::size_t cell, int subdivisions = 64) const {
        std::vector<double> cdf(static_cast<std::size_t>(subdivisions + 1), 0.0);
        const double left = mesh_.edge[cell];
        const double right = mesh_.edge[cell + 1];
        long double sum = 0.0L;
        for (int q = 0; q < subdivisions; ++q) {
            const double a = left + (right - left) * q / subdivisions;
            const double b = left + (right - left) * (q + 1) / subdivisions;
            sum += density_scale_ * integrate_shape_mass(a, b);
            cdf[static_cast<std::size_t>(q + 1)] = static_cast<double>(sum);
        }
        const double normalization = cdf.back();
        for (double& value : cdf) {
            value /= normalization;
        }
        cdf.back() = 1.0;
        return cdf;
    }

private:
    double shape(double radius) const {
        const double core = 0.01;
        const double taper = 0.5 * (1.0 - std::tanh(radius - 9.0));
        return std::pow(radius * radius + core * core, -0.5 * gamma_) * taper;
    }

    long double integrate_shape_mass(double left, double right) const {
        // Map the twelve Gauss nodes from [-1,1] into this radial interval
        // and integrate the spherical mass measure 4*pi*r^2*rho dr.
        const auto& rule = ev::phase_space_gauss_legendre_rule(12);
        const double midpoint = 0.5 * (left + right);
        const double half_width = 0.5 * (right - left);
        long double result = 0.0L;
        for (std::size_t q = 0; q < rule.first.size(); ++q) {
            const double radius = midpoint + half_width * rule.first[q];
            result += static_cast<long double>(rule.second[q])
                * 4.0L * ev::PI * radius * radius * shape(radius);
        }
        return result * half_width;
    }

    double gamma_ = 0.0;
    const ev::WeightedPICRadialMesh& mesh_;
    double density_scale_ = 0.0;
    double outer_mass_ = 0.0;
    double central_mass_ = 0.0;
    std::vector<double> shell_mass_;
    std::vector<double> enclosed_mass_at_left_;
};

// Convert a uniform probability into a standard normal variate. Separate
// rational approximations resolve the central distribution and its two tails.
double inverse_normal_cdf(double probability) {
    if (!(probability > 0.0 && probability < 1.0)) {
        throw std::runtime_error("Normal quantile requires 0 < p < 1.");
    }
    constexpr double a[] = {
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00};
    constexpr double b[] = {
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01};
    constexpr double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00};
    constexpr double d[] = {
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};
    constexpr double lower = 0.02425;
    constexpr double upper = 1.0 - lower;
    if (probability < lower) {
        const double q = std::sqrt(-2.0 * std::log(probability));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4])
                    * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (probability > upper) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - probability));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4])
                     * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const double q = probability - 0.5;
    const double r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4])
                * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4])
                * r + 1.0);
}

// Integral of x exp[-(x-a)^2/2] from zero to x. Here x=L/width and
// a=center/width; the common width^2 factor cancels from the normalized CDF.
// The initializer has 0 <= a <= 12.5. erfc preserves a distribution whose
// allowed angular-momentum interval lies deep below its Gaussian center.
long double tangential_cumulative(long double x, long double a) {
    if (!(x > 0.0L)) return 0.0L;
    if (a == 0.0L) return -std::expm1(-0.5L * x * x);
    const long double at_zero = std::exp(-0.5L * a * a);
    if (x * (1.0L + a) < 0.1L) {
        // Near the origin the two analytic terms cancel at linear order.
        // Eight-point Gauss--Legendre quadrature of the positive integrand
        // resolves this short, smoothly varying interval to roundoff.
        constexpr long double nodes[] = {
            0.18343464249564980494L, 0.52553240991632898582L,
            0.79666647741362673959L, 0.96028985649753623168L};
        constexpr long double weights[] = {
            0.36268378337836198297L, 0.31370664587788728734L,
            0.22238103445337447054L, 0.10122853629037625915L};
        long double sum = 0.0L;
        for (int i = 0; i < 4; ++i) {
            for (int sign : {-1, 1}) {
                const long double point = 0.5L * x * (1.0L + sign * nodes[i]);
                sum += weights[i] * point
                    * std::exp(a * point - 0.5L * point * point);
            }
        }
        return 0.5L * x * at_zero * sum;
    }
    const long double z = x - a;
    const long double sqrt_two = std::sqrt(2.0L);
    const long double gaussian_integral = z <= 0.0L
        ? std::erfc(-z / sqrt_two) - std::erfc(a / sqrt_two)
        : std::erf(z / sqrt_two) + std::erf(a / sqrt_two);
    const long double exponential_difference = -at_zero
        * std::expm1(a * x - 0.5L * x * x);
    return exponential_difference
        + a * std::sqrt(0.5L * std::acos(-1.0L)) * gaussian_integral;
}

// Invert the normalized L-weighted Gaussian integral on the allowed interval.
// Bisection preserves the physical lower and upper angular-momentum limits.
double sample_angular_momentum(
    double probability,
    double center,
    double width,
    double maximum) {
    const long double scaled_center = static_cast<long double>(center) / width;
    const long double normalization = tangential_cumulative(
        static_cast<long double>(maximum) / width, scaled_center);
    if (!(normalization > 0.0L) || !std::isfinite(normalization)) {
        throw std::runtime_error("Angular distribution has invalid normalization.");
    }
    const long double target = probability * normalization;
    double left = 0.0;
    double right = maximum;
    for (int iteration = 0; iteration < 48; ++iteration) {
        const double midpoint = 0.5 * (left + right);
        if (tangential_cumulative(
                static_cast<long double>(midpoint) / width, scaled_center) < target) {
            left = midpoint;
        } else {
            right = midpoint;
        }
    }
    return 0.5 * (left + right);
}

// Locate the cumulative-probability interval, then interpolate within it.
// This maps a uniform draw to the sampled coordinate with the prescribed CDF.
double invert_piecewise_cdf(
    double probability,
    double left,
    double right,
    const std::vector<double>& cdf) {
    const auto upper = std::upper_bound(cdf.begin(), cdf.end(), probability);
    const std::size_t hi = std::clamp<std::size_t>(
        static_cast<std::size_t>(upper - cdf.begin()), 1, cdf.size() - 1);
    const std::size_t lo = hi - 1;
    const double fraction = (probability - cdf[lo])
        / std::max(cdf[hi] - cdf[lo], 1.0e-300);
    const double logical = (static_cast<double>(lo) + fraction)
        / static_cast<double>(cdf.size() - 1);
    return left + logical * (right - left);
}

// Construct positive-mass particles shell by shell. Opposite radial-velocity
// deviations suppress a spurious mean inflow. Optional angular importance
// sampling uses more particles in a small-L interval while assigning that
// interval its original probability mass. Each shell retains its own total mass.
std::vector<ev::WeightedPICMarker> initialize_markers(
    const Options& run,
    const ev::Params& p,
    const ev::WeightedPICFieldGrid& grid,
    const PowerLawInitialCondition& initial) {
    const bool use_angular_importance =
        run.low_angular_tail_probability > 0.0;
    auto source_uses_angular_importance = [&](double source_right_edge) {
        return use_angular_importance
            && (run.low_angular_source_rmax == 0.0
                || source_right_edge <= run.low_angular_source_rmax
                    * (1.0 + 64.0 * std::numeric_limits<double>::epsilon()));
    };
    std::size_t marker_reserve = 0;
    marker_reserve += static_cast<std::size_t>(
        run.particles_per_radial_cell
        * (source_uses_angular_importance(grid.mesh.edge.front())
                ? run.low_angular_source_ppc_multiplier
                : 1));
    for (int cell = 0; cell < p.Nr; ++cell) {
        if (!(initial.shell_mass(static_cast<std::size_t>(cell)) > 0.8e-14)) {
            continue;
        }
        const bool targeted = source_uses_angular_importance(
            grid.mesh.edge[static_cast<std::size_t>(cell + 1)]);
        marker_reserve += static_cast<std::size_t>(
            run.particles_per_radial_cell
            * (targeted ? run.low_angular_source_ppc_multiplier : 1));
    }
    std::vector<ev::WeightedPICMarker> markers;
    markers.reserve(marker_reserve);
    const int base_pairs = run.particles_per_radial_cell / 2;
    std::uint64_t id = 0;
    auto append_quiet_pair = [&](double radius,
                                 double enclosed_mass,
                                 std::uint64_t source_cell,
                                 std::uint64_t sampling_source,
                                 int pair,
                                 int stratum_pairs,
                                 double angular_probability_left,
                                 double angular_probability_right,
                                 std::uint64_t angular_stream,
                                 double marker_mass) {
        // Balance the enclosed gravitational acceleration and prescribed
        // background acceleration when choosing the initial tangential speed.
        const double expansion_support = p.vp_alpha * (1.0 - p.vp_alpha)
            * radius * radius / (p.vp_time_offset * p.vp_time_offset);
        const double tangential_speed = std::sqrt(std::max(
            enclosed_mass / radius + expansion_support, 1.0e-300));
        const double velocity_width = std::max(0.08 * tangential_speed, 0.015);
        const double tangential_width = radius * velocity_width;
        const double angular_center = radius * tangential_speed;
        // Use the lower half of a Gaussian CDF, then pair opposite deviations
        // about zero physical radial velocity to keep the radial mean quiet.
        const double normal_probability =
            (static_cast<double>(pair) + 0.5)
            / static_cast<double>(2 * stratum_pairs);
        const double normal = inverse_normal_cdf(normal_probability);
        for (int sign = 0; sign < 2; ++sign) {
            const std::uint64_t ordinal =
                static_cast<std::uint64_t>(2 * pair + sign);
            const std::uint64_t angular_counter = angular_stream == 0
                ? sampling_source
                : sampling_source
                    ^ ev::weighted_pic_mix64(angular_stream + 1);
            const double unit_angular_probability = std::fmod(
                ev::weighted_pic_radical_inverse(ordinal + 1, 3)
                    + ev::weighted_pic_counter_uniform(
                        UINT64_C(0x706f7765726c6177),
                        angular_counter,
                        static_cast<std::uint64_t>(100.0 * run.gamma)
                            + UINT64_C(0x10000) * angular_stream),
                1.0);
            // Transform a low-discrepancy sample into the selected angular
            // probability interval before inverting its momentum CDF.
            const double angular_probability =
                angular_probability_left
                + (angular_probability_right - angular_probability_left)
                    * unit_angular_probability;
            const double angular_momentum = sample_angular_momentum(
                angular_probability,
                angular_center,
                tangential_width,
                p.L_max);
            ev::WeightedPICMarker marker;
            marker.mass = marker_mass;
            marker.r = radius;
            // Subtract H*r to store peculiar rather than physical velocity.
            marker.w = -p.vp_alpha * radius / p.vp_time_offset
                + (sign == 0 ? normal : -normal) * velocity_width;
            marker.eta = 0.5 * angular_momentum * angular_momentum;
            if (run.initial_velocity_factor != 1.0) {
                // w is peculiar velocity. Rescale physical motion about the
                // halo center, then convert back using the same background.
                const double hubble_velocity = p.vp_alpha * radius
                    / p.vp_time_offset;
                marker.w = run.initial_velocity_factor
                    * (marker.w + hubble_velocity) - hubble_velocity;
                marker.eta *= run.initial_velocity_factor
                    * run.initial_velocity_factor;
            }
            marker.id = id++;
            marker.source_cell = source_cell;
            markers.push_back(marker);
        }
    };
    auto append_angular_strata = [&](
        int population_pairs,
        bool importance_targeted,
        auto&& append_stratum) {
        if (!importance_targeted) {
            append_stratum(
                population_pairs,
                0.0,
                1.0,
                1.0,
                UINT64_C(0));
            return;
        }
        // Split sampling effort between the low-L tail and the rest. The
        // mass fractions passed below remain the physical CDF fractions,
        // independently of how many pairs sample either interval.
        const int tail_pairs = std::clamp(
            static_cast<int>(std::llround(
                run.low_angular_tail_fraction * population_pairs)),
            1,
            population_pairs - 1);
        const int bulk_pairs = population_pairs - tail_pairs;
        append_stratum(
            tail_pairs,
            0.0,
            run.low_angular_tail_probability,
            run.low_angular_tail_probability,
            UINT64_C(1));
        append_stratum(
            bulk_pairs,
            run.low_angular_tail_probability,
            1.0,
            1.0 - run.low_angular_tail_probability,
            UINT64_C(2));
    };

    // The original origin-parity solver closes the unresolved sphere with the
    // first resolved density.  Represent that same uniform core with active
    // markers.  Below r_min, each marker cloud is handed continuously between
    // the first resolved shell and the hidden Poisson carrier in q=r^3/3; it
    // becomes wholly hidden only at the inner end of that transition.  The
    // marker can therefore cross the resolved face in either direction
    // without a wall or a discontinuous gravity impulse.
    const double core_radius = grid.mesh.edge.front();
    const bool core_is_targeted =
        source_uses_angular_importance(core_radius);
    const int core_pairs = base_pairs
        * (core_is_targeted
                ? run.low_angular_source_ppc_multiplier
                : 1);
    append_angular_strata(
        core_pairs,
        core_is_targeted,
        [&](int stratum_pairs,
            double angular_probability_left,
            double angular_probability_right,
            double represented_mass_fraction,
            std::uint64_t angular_stream) {
            const double core_marker_mass =
                initial.central_mass() * represented_mass_fraction
                / static_cast<double>(2 * stratum_pairs);
            for (int pair = 0; pair < stratum_pairs; ++pair) {
                const double probability =
                    (static_cast<double>(pair) + 0.5) / stratum_pairs;
                const double radius = core_radius * std::cbrt(probability);
                append_quiet_pair(
                    radius,
                    probability * initial.central_mass(),
                    0,
                    static_cast<std::uint64_t>(p.Nr),
                    pair,
                    stratum_pairs,
                    angular_probability_left,
                    angular_probability_right,
                    angular_stream,
                    core_marker_mass);
            }
        });
    const std::size_t first_resolved_marker = markers.size();
    double represented_mass = 0.0;
    for (int cell = 0; cell < p.Nr; ++cell) {
        const double shell_mass = initial.shell_mass(static_cast<std::size_t>(cell));
        if (!(shell_mass > 0.8e-14)) {
            continue;
        }
        const std::vector<double> radial_cdf =
            initial.radial_cdf(static_cast<std::size_t>(cell));
        const double left = grid.mesh.edge[static_cast<std::size_t>(cell)];
        const double right = grid.mesh.edge[static_cast<std::size_t>(cell + 1)];
        const bool cell_is_targeted =
            source_uses_angular_importance(right);
        const int cell_pairs = base_pairs
            * (cell_is_targeted
                    ? run.low_angular_source_ppc_multiplier
                    : 1);
        append_angular_strata(
            cell_pairs,
            cell_is_targeted,
            [&](int stratum_pairs,
                double angular_probability_left,
                double angular_probability_right,
                double represented_mass_fraction,
                std::uint64_t angular_stream) {
                // Divide this stratum mass by its actual particle count.
                // Shell-dependent masses follow from the prescribed density.
                const double marker_mass =
                    shell_mass * represented_mass_fraction
                    / static_cast<double>(2 * stratum_pairs);
                for (int pair = 0; pair < stratum_pairs; ++pair) {
                    const double radial_probability =
                        (static_cast<double>(pair) + 0.5) / stratum_pairs;
                    const double radius = invert_piecewise_cdf(
                        radial_probability, left, right, radial_cdf);
                    const double enclosed_mass = initial.enclosed_mass_at_left(
                            static_cast<std::size_t>(cell))
                        + radial_probability * shell_mass;
                    append_quiet_pair(
                        radius,
                        enclosed_mass,
                        static_cast<std::uint64_t>(cell),
                        static_cast<std::uint64_t>(cell),
                        pair,
                        stratum_pairs,
                        angular_probability_left,
                        angular_probability_right,
                        angular_stream,
                        marker_mass);
                }
            });
        represented_mass += shell_mass;
    }
    // Correct the tiny mass omitted by the shell activity threshold with
    // one common factor on resolved particles. The inner mass stays separate.
    const double correction = initial.outer_mass() / represented_mass;
    for (std::size_t q = first_resolved_marker; q < markers.size(); ++q) {
        markers[q].mass *= correction;
    }
    for (ev::WeightedPICMarker& marker : markers) {
        ev::validate_weighted_pic_marker(marker, "Power-law WPIC initialization");
    }
    return markers;
}

// ---- Dynamical limits, energy sums and virial diagnostics ------------------
double choose_timestep(
    const ev::WeightedPICDepositedMoments& deposit,
    const ev::VlasovPoissonField& field,
    const ev::WeightedPICRadialMesh& mesh,
    const Options& run) {
    // Collisionless KDK deliberately uses one fixed orbit step (apart from
    // shorter landings on output and final times).  Time-symmetric fixed-step
    // KDK gives the desired bounded energy error in static-potential controls.
    // Collisional runs retain the dynamical orbit limiter because their
    // independently scheduled Boltzmann events break that symmetry.
    if (run.subhalo_collision_strength == 0.0) {
        return run.maximum_timestep;
    }
    double timestep = run.maximum_timestep;
    const double mass_floor = 1.0e-12 * deposit.deposited_mesh_mass;
    for (std::size_t cell = 0; cell < deposit.shell_mass.size(); ++cell) {
        const double mass = deposit.shell_mass[cell];
        if (!(mass > mass_floor)) {
            continue;
        }
        const double radius = mesh.venus_center[cell];
        // The deposited second moment is extensive: divide by shell mass
        // to obtain <v_t^2>. The inverse frequency below resolves gravity,
        // angular motion and background expansion in the collisional branch.
        const double tangential_second =
            deposit.shell_tangential_second_moment[cell] / mass;
        const double omega_squared = std::abs(field.phi_r[cell]) / radius
            + tangential_second / (radius * radius)
            + field.background.H * field.background.H;
        if (omega_squared > 0.0 && std::isfinite(omega_squared)) {
            timestep = std::min(
                timestep, run.dynamical_fraction / std::sqrt(omega_squared));
        }
    }
    if (!(timestep > 0.0) || !std::isfinite(timestep)) {
        throw std::runtime_error("Expanding WPIC selected an invalid timestep.");
    }
    return timestep;
}

/// Shortest spherical free-fall/orbital clock on the represented mass grid.
///
/// For enclosed physical mass M(<r), the gravitational frequency is
/// omega_dyn^2=G M(<r)/r^3 and t_dyn=1/omega_dyn.  This deliberately uses
/// total matter, rather than the density-contrast Poisson source, because a
/// locally bound subhalo responds to its actual enclosed mass.
double minimum_dynamical_timescale(
    const ev::WeightedPICDepositedMoments& deposit,
    const ev::WeightedPICRadialMesh& mesh,
    const ev::Params& p) {
    long double enclosed_mass =
        static_cast<long double>(deposit.hidden_inner_mass + p.central_mass);
    const double mass_floor =
        1.0e-14 * std::max(1.0, deposit.deposited_mesh_mass);
    double minimum = std::numeric_limits<double>::infinity();
    for (std::size_t cell = 0; cell < deposit.shell_mass.size(); ++cell) {
        // Hidden interior mass and every inner shell contribute to the
        // acceleration at this radius, even if the present shell is dilute.
        enclosed_mass += deposit.shell_mass[cell];
        const double radius = mesh.venus_center[cell];
        if (!(enclosed_mass > mass_floor) || !(radius > 0.0)) {
            continue;
        }
        const long double frequency_squared =
            static_cast<long double>(p.vp_gravity) * enclosed_mass
            / (static_cast<long double>(radius) * radius * radius);
        if (frequency_squared > 0.0L
            && std::isfinite(static_cast<double>(frequency_squared))) {
            minimum = std::min(
                minimum,
                1.0 / std::sqrt(static_cast<double>(frequency_squared)));
        }
    }
    return minimum;
}

/// Shortest measured time on which the represented potential changes.
///
/// Successive self-consistent Poisson solves provide finite-difference
/// estimates of Phi_dot and g_dot, where g=-dPhi/dr.  We use the faster of
/// ||Phi||_infinity/||Phi_dot||_infinity and
/// ||g||_infinity/||g_dot||_infinity because the potential sets particle
/// energy while its gradient directly accelerates the markers.  Global
/// maximum norms avoid a discontinuous ratio where Phi or g crosses zero.
///
/// The field is a finite-marker estimate.  Dividing a sub-noise deposition
/// residual by a near-zero cleanup dt would manufacture an arbitrarily short
/// t_phi and force an endless cadence-halving loop.  The effective marker
/// number therefore supplies a conservative sampling floor; only field
/// changes resolved above that floor enter the derivative clock.
double minimum_potential_change_timescale(
    const ev::VlasovPoissonField& previous,
    const ev::VlasovPoissonField& current,
    double dt,
    const ev::WeightedPICDepositedMoments& current_deposit) {
    if (!(dt > 0.0)
        || previous.phi.size() != current.phi.size()
        || previous.phi_r.size() != current.phi_r.size()
        || previous.phi.size() != previous.phi_r.size()
        || current_deposit.shell_mass.size()
            != current_deposit.effective_marker_number.size()) {
        return std::numeric_limits<double>::infinity();
    }
    long double deposited_mass = 0.0L;
    long double squared_marker_mass = 0.0L;
    for (std::size_t cell = 0;
         cell < current_deposit.shell_mass.size();
         ++cell) {
        const double shell_mass = current_deposit.shell_mass[cell];
        const double shell_neff =
            current_deposit.effective_marker_number[cell];
        deposited_mass += shell_mass;
        if (shell_mass > 0.0 && shell_neff > 0.0) {
            squared_marker_mass +=
                static_cast<long double>(shell_mass) * shell_mass
                / shell_neff;
        }
    }
    if (!(deposited_mass > 0.0L)
        || !(squared_marker_mass > 0.0L)) {
        return std::numeric_limits<double>::infinity();
    }
    // (sum m)^2/sum(m^2) measures the statistical size of an unequal-mass
    // sample. Use it to distinguish an evolving field from deposition noise.
    const double effective_marker_number = static_cast<double>(
        deposited_mass * deposited_mass / squared_marker_mass);
    return ev::weighted_pic_noise_aware_field_change_clock(
        previous.phi,
        previous.phi_r,
        current.phi,
        current.phi_r,
        dt,
        effective_marker_number).timescale;
}

// Sum peculiar translational kinetic energy, half the sampled gravitational
// potential energy, and the subhalo internal energy.  The half factor counts
// self-gravitating pairs once.  In an expanding background this sum is a
// diagnostic whose evolution also depends on the background terms.
double total_energy(
    const std::vector<ev::WeightedPICMarker>& markers,
    const ev::VlasovPoissonField& field,
    const ev::WeightedPICFieldGrid& grid,
    const ev::Params& p) {
    long double energy = 0.0L;
    for (const ev::WeightedPICMarker& marker : markers) {
        if (marker.state != ev::WeightedPICMarkerState::active) {
            continue;
        }
        const long double radius = std::max(marker.r, 1.0e-150);
        const long double tangential_second = 2.0L * marker.eta
            / (radius * radius);
        const ev::VPSample sample = ev::sample_vp_field(
            marker.r,
            grid.radial_coordinate,
            grid.radius,
            field,
            p);
        energy += 0.5L * marker.mass
            * (marker.w * marker.w + tangential_second + sample.phi)
            + marker.internal_energy;
    }
    return static_cast<double>(energy);
}

/// Global marker moments for the scalar virial identity between encounters.
/// I=sum(m*r^2), C=sum(m*r*w) obey I_dot=2*H*I+2*C and
/// C_dot=2*K_pec+W_force, with W_force=-sum(m*r*Phi_r). Sampling the same
/// field as the pusher includes the unresolved inner carrier and background
/// source consistently. Wall impulses and finite collision jumps are separate
/// contributions when differencing these moments through such events.
struct MarkerVirialBudget {
    long double mass = 0.0L;
    long double inertia = 0.0L;
    long double radial_moment = 0.0L;
    long double kinetic_peculiar = 0.0L;
    long double kinetic_physical = 0.0L;
    long double force_virial = 0.0L;
    long double internal_energy = 0.0L;
};

MarkerVirialBudget marker_virial_budget(
    const std::vector<ev::WeightedPICMarker>& markers,
    const ev::VlasovPoissonField& field,
    const ev::WeightedPICFieldGrid& grid,
    const ev::Params& p) {
    MarkerVirialBudget result;
    for (const auto& marker : markers) {
        if (marker.state != ev::WeightedPICMarkerState::active) continue;
        const long double mass = marker.mass;
        const long double r = std::max(marker.r, 1.0e-150);
        const long double w = marker.w;
        const long double vt2 = 2.0L * marker.eta / (r * r);
        const long double u = field.background.H * r + w;
        const auto sample = ev::sample_vp_field(
            marker.r, grid.radial_coordinate, grid.radius, field, p);
        result.mass += mass;
        result.inertia += mass * r * r;
        result.radial_moment += mass * r * w;
        result.kinetic_peculiar += 0.5L * mass * (w * w + vt2);
        result.kinetic_physical += 0.5L * mass * (u * u + vt2);
        result.force_virial -= mass * r * sample.phi_r;
        result.internal_energy += marker.internal_energy;
    }
    return result;
}

// ---- Profiles, metadata, and atomic checkpoints ----------------------------
// Convert the deposited raw moments into the mean radial velocity and its
// variance: <w^2>-<w>^2.  Keep the tangential second moment as a separate
// column.  Each row uses the same shell density and effective sample number.
void write_profile(
    const std::filesystem::path& directory,
    std::int64_t output_index,
    double time,
    const ev::WeightedPICDepositedMoments& deposit,
    const ev::WeightedPICRadialMesh& mesh) {
    std::ostringstream filename;
    filename << "profile_" << std::setw(5) << std::setfill('0')
             << output_index << ".csv";
    const std::filesystem::path target = directory / filename.str();
    const std::filesystem::path temporary = target.string() + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Unable to open temporary weighted-PIC profile: "
            + temporary.string());
    }
    output << std::setprecision(17)
           << "time,radius,density,radial_mean,radial_variance,"
              "tangential_second,neff\n";
    for (std::size_t cell = 0; cell < deposit.density.size(); ++cell) {
        const double density = deposit.density[cell];
        const double mean = density > 0.0
            ? deposit.radial_momentum_density[cell] / density : 0.0;
        const double radial_second = density > 0.0
            ? deposit.radial_second_moment_density[cell] / density : 0.0;
        const double tangential_second = density > 0.0
            ? deposit.tangential_second_moment_density[cell] / density : 0.0;
        output << time << ',' << mesh.venus_center[cell] << ',' << density << ','
               << mean << ',' << std::max(0.0, radial_second - mean * mean)
               << ',' << tangential_second << ','
               << deposit.effective_marker_number[cell] << '\n';
    }
    output.flush();
    if (!output) {
        throw std::runtime_error(
            "Unable to complete temporary weighted-PIC profile: "
            + temporary.string());
    }
    output.close();
    if (!output) {
        std::filesystem::remove(temporary);
        throw std::runtime_error(
            "Unable to close temporary weighted-PIC profile: "
            + temporary.string());
    }

    // A monitor must never mistake a partially written final-name CSV for a
    // complete radial profile.  Rename within the output directory only after
    // every radial row has been written and the staging stream has closed.
    // The replacement branch
    // supports deterministic restart replay over a profile written beyond the
    // last committed checkpoint.
    std::error_code error;
    std::filesystem::rename(temporary, target, error);
    if (error) {
        std::filesystem::remove(target, error);
        error.clear();
        std::filesystem::rename(temporary, target, error);
    }
    if (error) {
        std::filesystem::remove(temporary);
        throw std::runtime_error(
            "Unable to install weighted-PIC profile atomically: "
            + target.string());
    }
}

// Store particles together with the encounter RNG, accumulated collision
// age and measured field/collision timescales.  A continuation needs these
// quantities to preserve the event sequence across the checkpoint.
void write_restart_atomic(
    const std::filesystem::path& directory,
    const std::vector<ev::WeightedPICMarker>& markers,
    double time,
    std::int64_t step,
    const ev::WeightedPICCollisionSamplerState& collision_sampler,
    double collision_event_age,
    double collision_timescale_estimate,
    double potential_timescale_estimate) {
    ev::WeightedPICRestartState state;
    state.markers = markers;
    state.time = time;
    state.step = step;
    state.sampler = {UINT64_C(0x706f7765726c6177), 0};
    state.collision_sampler = collision_sampler;
    state.collision_event_age = collision_event_age;
    state.collision_timescale_estimate = collision_timescale_estimate;
    state.potential_timescale_estimate = potential_timescale_estimate;
    const std::filesystem::path temporary = directory / "restart_state.tmp";
    const std::filesystem::path target = directory / "restart_state.bin";
    ev::write_weighted_pic_restart(temporary, state);
    std::error_code error;
    std::filesystem::rename(temporary, target, error);
    if (error) {
        std::filesystem::remove(target, error);
        error.clear();
        std::filesystem::rename(temporary, target, error);
    }
    if (error) {
        throw std::runtime_error("Could not atomically replace the WPIC restart.");
    }
}

/// Persist the fixed radial mesh as a small restart-side contract.
///
/// Marker checkpoints store particle data separately from the field mesh.
/// This sidecar binds the saved state to the radial cells and domain on which
/// its field and collision diagnostics were measured.
void write_radial_mesh_contract_atomic(
    const std::filesystem::path& directory,
    const ev::Params& p) {
    const std::filesystem::path temporary =
        directory / "radial_mesh_contract.tmp";
    const std::filesystem::path target =
        directory / "radial_mesh_contract.txt";
    {
        std::ofstream output(temporary);
        if (!output) {
            throw std::runtime_error(
                "Could not create the WPIC radial-mesh contract.");
        }
        output << std::setprecision(17)
               << "VENUS_WPIC_RADIAL_MESH 1\n"
               << "radial_cells " << p.Nr << '\n'
               << "radial_minimum " << p.r_min << '\n'
               << "radial_maximum " << p.r_max << '\n';
        if (!output) {
            throw std::runtime_error(
                "Could not write the WPIC radial-mesh contract.");
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, target, error);
    if (error) {
        std::filesystem::remove(target, error);
        error.clear();
        std::filesystem::rename(temporary, target, error);
    }
    if (error) {
        throw std::runtime_error(
            "Could not atomically replace the WPIC radial-mesh contract.");
    }
}

/// Require a restart to use the same mesh on which its fields were measured.
///
/// Checkpoints without a sidecar are accepted only on the fixed 256-cell
/// [0.1,1000] mesh specified by that restart format.  Once resumed, they
/// receive explicit mesh metadata and all later continuations are strict.
void validate_restart_radial_mesh(
    const std::filesystem::path& restart,
    const ev::Params& p) {
    const std::filesystem::path contract =
        restart.parent_path() / "radial_mesh_contract.txt";
    if (!std::filesystem::exists(contract)) {
        const bool legacy_default_mesh =
            p.Nr == 256 && p.r_min == 0.1 && p.r_max == 1000.0;
        if (!legacy_default_mesh) {
            throw std::runtime_error(
                "A WPIC restart without radial_mesh_contract.txt "
                "can only use --radial-cells 256 --radial-minimum 0.1 "
                "--radial-maximum 1000.");
        }
        return;
    }

    std::ifstream input(contract);
    std::string magic;
    int version = 0;
    std::string cells_label;
    int radial_cells = 0;
    std::string minimum_label;
    double radial_minimum = 0.0;
    std::string maximum_label;
    double radial_maximum = 0.0;
    input >> magic >> version
          >> cells_label >> radial_cells
          >> minimum_label >> radial_minimum
          >> maximum_label >> radial_maximum;
    if (!input || magic != "VENUS_WPIC_RADIAL_MESH" || version != 1
        || cells_label != "radial_cells"
        || minimum_label != "radial_minimum"
        || maximum_label != "radial_maximum") {
        throw std::runtime_error(
            "Malformed WPIC radial_mesh_contract.txt beside restart.");
    }
    if (radial_cells != p.Nr
        || radial_minimum != p.r_min
        || radial_maximum != p.r_max) {
        throw std::runtime_error(
            "WPIC restart radial mesh differs from the requested mesh.");
    }
}

void write_metadata(
    const Options& run,
    const ev::Params& p,
    const PowerLawInitialCondition& initial,
    const std::vector<ev::WeightedPICMarker>& markers) {
    const bool analytic_powerlaw = run.initial_condition_label == "powerlaw";
    ev::weighted_pic_detail::CompensatedMassSum total_mass;
    ev::weighted_pic_detail::CompensatedMassSum inner_mass;
    if (!analytic_powerlaw) {
        for (const auto& marker : markers) {
            total_mass.add(marker.mass);
            if (marker.r < p.r_min) inner_mass.add(marker.mass);
        }
    }
    std::ofstream output(run.output_directory / "metadata.json");
    output << std::setprecision(17)
           << "{\n"
           << "  \"model\": \""
           << (run.subhalo_collision_strength > 0.0
                   ? "weighted_pic_subhalo_scattering_capture_"
                   : "weighted_pic_collisionless_")
           << run.initial_condition_label << "\",\n"
           << "  \"gamma_i\": ";
    if (analytic_powerlaw) output << run.gamma;
    else output << "null";
    output << ",\n"
           << "  \"vp_universe\": \"" << p.vp_universe << "\",\n"
           << "  \"vp_alpha\": " << p.vp_alpha << ",\n"
           << "  \"vp_time_offset\": " << p.vp_time_offset << ",\n"
           << "  \"initial_velocity_factor\": "
           << run.initial_velocity_factor << ",\n"
           << "  \"initial_velocity_convention\": \""
           << (analytic_powerlaw
               ? "fresh physical velocity u=H*r+w scaled once; eta scaled by factor squared"
               : "native particle velocities retained; physical u=H*r+w")
           << "\",\n"
           << "  \"vp_poisson_source\": \"density_contrast\",\n"
           << "  \"radial_cells\": " << p.Nr << ",\n"
           << "  \"radial_minimum\": " << p.r_min << ",\n"
           << "  \"radial_maximum\": " << p.r_max << ",\n"
           << "  \"particles_per_radial_cell\": "
           << run.particles_per_radial_cell << ",\n"
           << "  \"low_angular_tail_probability\": "
           << run.low_angular_tail_probability << ",\n"
           << "  \"low_angular_tail_fraction\": "
           << run.low_angular_tail_fraction << ",\n"
           << "  \"low_angular_source_rmax\": "
           << run.low_angular_source_rmax << ",\n"
           << "  \"low_angular_source_ppc_multiplier\": "
           << run.low_angular_source_ppc_multiplier << ",\n"
           << "  \"marker_count\": " << markers.size() << ",\n";
    if (analytic_powerlaw) {
        output << "  \"resolved_halo_mass\": " << initial.outer_mass() << ",\n"
               << "  \"initial_origin_core_mass\": " << initial.central_mass() << ",\n"
               << "  \"mobile_mass\": "
               << initial.outer_mass() + initial.central_mass() << ",\n";
    } else {
        output << "  \"initial_condition_label\": \""
               << run.initial_condition_label << "\",\n"
               << "  \"native_input_sha256\": \""
               << ev::integrity::sha256_file(run.restart) << "\",\n"
               << "  \"marker_mass_at_start\": "
               << static_cast<double>(total_mass.value()) << ",\n"
               << "  \"marker_mass_inside_inner_face_at_start\": "
               << static_cast<double>(inner_mass.value()) << ",\n"
               << "  \"marker_mass_in_resolved_domain_at_start\": "
               << static_cast<double>(total_mass.value() - inner_mass.value()) << ",\n"
               << "  \"mobile_mass\": "
               << static_cast<double>(total_mass.value()) << ",\n";
    }
    output
           << "  \"inner_boundary\": \"origin_parity\",\n"
           << "  \"end_time\": " << run.end_time << ",\n"
           << "  \"maximum_timestep\": " << run.maximum_timestep << ",\n"
           << "  \"dynamical_fraction\": " << run.dynamical_fraction << ",\n"
           << "  \"output_interval\": " << run.output_interval << ",\n"
           << "  \"history_every\": " << run.history_every << ",\n"
           << "  \"restart_every\": " << run.restart_every << ",\n"
           << "  \"output_directory\": \""
           << run.output_directory.string() << "\",\n"
           << "  \"restart_source\": \""
           << run.restart.string() << "\",\n"
           << "  \"subhalo_collision_strength\": "
           << run.subhalo_collision_strength << ",\n"
           << "  \"subhalo_velocity_scale\": "
           << run.subhalo_velocity_scale << ",\n"
           << "  \"subhalo_velocity_scale_role\": "
              "\"initial_marker_internal_dispersion\",\n"
           << "  \"subhalo_internal_dispersion_evolves\": "
           << (run.subhalo_collision_strength > 0.0 ? "true" : "false")
           << ",\n"
           << "  \"subhalo_pair_velocity_scale\": "
              "\"sqrt((sqrt(m_i)*sigma_i^2+sqrt(m_j)*sigma_j^2)"
              "/(sqrt(m_i)+sqrt(m_j)))\",\n"
           << "  \"subhalo_internal_energy_relation\": "
              "\"E_int=(3/2)*m*sigma_s^2\",\n"
           << "  \"subhalo_plummer_area_per_mass\": "
           << run.subhalo_plummer_area_per_mass << ",\n"
           << "  \"subhalo_relative_normalization\": "
              "\"tabulated_exclusive_plummer_rp_le_2a\",\n"
           << "  \"subhalo_low_speed_regularization\": "
           << run.subhalo_low_speed_regularization << ",\n"
           << "  \"subhalo_collision_interval\": "
           << run.subhalo_collision_interval << ",\n"
           << "  \"subhalo_collision_interval_role\": "
           << (run.subhalo_adaptive_cell_timestep
                   ? "\"maximum; cell-local CFL handles collision rates, "
                     "while the batch follows dynamical and "
                     "potential-change clocks\",\n"
                   : "\"maximum; adaptive interval is epsilon_op times the "
                     "minimum collision/dynamical/potential-change "
                     "timescale\",\n")
           << "  \"subhalo_collision_timescale_fraction\": "
           << run.subhalo_collision_timescale_fraction << ",\n"
           << "  \"subhalo_collision_endpoint_transactional_retry\": "
           << (
               run.subhalo_collision_strength > 0.0
                   && run.subhalo_collision_timescale_fraction > 0.0
               ? "true" : "false")
           << ",\n"
           << "  \"subhalo_collision_cadence_stop_exit_status\": 97,\n"
           << "  \"potential_change_sampling_floor\": "
              "\"max(4096*machine_epsilon,0.25/sqrt(global_N_eff))\",\n"
           << "  \"subhalo_collision_cfl\": "
           << run.subhalo_collision_cfl << ",\n"
           << "  \"subhalo_collision_maximum_substeps\": "
           << run.subhalo_collision_maximum_substeps << ",\n"
           << "  \"subhalo_adaptive_cell_timestep\": "
           << (run.subhalo_adaptive_cell_timestep ? "true" : "false")
           << ",\n"
           << (run.collision_prestep_coordinate_due
                   ? "  \"collision_prestep_coordinate_due\": true,\n" : "")
           << "  \"subhalo_locality_maximum_ratio\": "
           << run.subhalo_locality_maximum_ratio << ",\n"
           << "  \"subhalo_inner_collision_cells\": "
           << run.subhalo_inner_collision_cells << ",\n"
           << "  \"subhalo_collision_lanes\": "
           << run.subhalo_collision_lanes << ",\n"
           << "  \"subhalo_pair_sampling\": \""
           << run.subhalo_pair_sampling << "\",\n"
           << "  \"subhalo_pair_proposal\": "
              "\"importance proportional to safe pair-specific "
              "u*(Sigma_scatt+Sigma_cap) majorant; exact kernel divided "
              "by the same bound at acceptance\",\n"
           << "  \"subhalo_collision_seed\": "
           << run.subhalo_collision_seed << ",\n"
           << "  \"subhalo_collision_lane_assignment\": "
              "\"epoch_hashed_marker_id_unbiased_ntc_stratification\",\n"
           << "  \"subhalo_repopulation_trigger_live_markers\": "
           << run.subhalo_repopulation_trigger_live_markers << ",\n"
           << "  \"subhalo_repopulation_refinement_factor\": "
           << run.subhalo_repopulation_refinement_factor << ",\n"
           << "  \"subhalo_repopulation_distribution\": "
              "\"disabled_fail_closed; legacy_exact_clone_fields_only\",\n"
           << "  \"subhalo_repopulation_physics\": "
              "\"positive_trigger_rejected_because_same_parent_daughters_add_"
              "spurious_collision_pairs\",\n"
           << "  \"subhalo_mean_free_path\": "
              "\"<u>/(n_partner*<u*Sigma_ij(u)>), with "
              "n_partner=(N-1)/V sampled from the evolving physical-cell "
              "distribution before lane subdivision\",\n"
           << "  \"subhalo_orbital_velocity_dispersion\": "
              "\"mass-weighted evolving local DF, "
              "sigma_v^2=(<v^2>-<v_r>^2)/3\",\n"
           << "  \"subhalo_inner_collision_cell_geometry\": "
              "\"equal_volume_0_to_radial_minimum\",\n"
           << "  \"subhalo_compression_trigger_markers\": "
           << run.subhalo_compression_trigger_markers << ",\n"
           << "  \"subhalo_compression_minimum_markers\": "
           << run.subhalo_compression_minimum_markers << ",\n"
           << "  \"subhalo_compression_phase_subcells\": "
           << run.subhalo_compression_phase_subcells << ",\n"
           << "  \"subhalo_compression_preserve_source_cell\": "
           << (run.subhalo_compression_preserve_source_cell
                   ? "true" : "false")
           << ",\n"
           << "  \"subhalo_compression_minimum_reduction_fraction\": "
           << run.subhalo_compression_minimum_reduction_fraction << ",\n"
           << "  \"subhalo_compression_tolerance\": "
           << run.subhalo_compression_tolerance << ",\n"
           << "  \"subhalo_compression_minimize_weight_variance\": "
           << (run.subhalo_compression_minimize_weight_variance
                   ? "true" : "false")
           << ",\n"
           << "  \"subhalo_compression_maximum_mass_amplification\": "
           << run.subhalo_compression_maximum_mass_amplification << ",\n"
           << "  \"threads\": " << run.threads << "\n"
           << "}\n";
}


/// Bind the expansion law to the checkpoint. Marker files store velocities,
/// so a background change would reinterpret the saved physical state.
void background_contract(const std::filesystem::path& directory,
                         const ev::Params& p, bool validate) {
    const auto path = directory / "background_contract.txt";
    std::ostringstream text;
    text << std::setprecision(17) << "VENUS_WPIC_BACKGROUND 1\n"
         << p.vp_universe << ' ' << p.vp_alpha << ' '
         << p.vp_time_offset << '\n';
    if (validate) {
        if (!std::filesystem::exists(path)) {
            // A checkpoint without a sidecar uses the prescribed expanding
            // background. Its continuation writes an explicit contract.
            if (p.vp_universe != "expanding" || p.vp_alpha != 2.0 / 3.0
                || p.vp_time_offset != 12.0) {
                throw std::runtime_error("A WPIC checkpoint requires "
                    "the expanding alpha=2/3, time-offset=12 background. "
                    "Fresh static/custom-background runs write their own contract.");
            }
            return;
        }
        std::ifstream input(path);
        std::ostringstream previous;
        previous << input.rdbuf();
        if (!input || previous.str() != text.str()) {
            throw std::runtime_error("WPIC restart background differs from "
                "background_contract.txt; repeat the original background options.");
        }
        return;
    }
    const auto temporary = directory / "background_contract.tmp";
    {
        std::ofstream output(temporary);
        output << text.str();
        output.flush();
        if (!output) throw std::runtime_error("Cannot write WPIC background contract.");
    }
    std::filesystem::rename(temporary, path);
}

/// Preserve the fresh distribution's label across deterministic restarts.
/// The factor is never applied to resumed particles. Repeating it authenticates
/// the initial-data provenance, while factor-one checkpoints remain
/// readable under their existing background and radial-mesh contracts.
void initial_condition_contract(const std::filesystem::path& directory,
                                const Options& run, bool validate) {
    const auto path = directory / "initial_condition_contract.txt";
    std::ostringstream text;
    text << std::setprecision(17) << "VENUS_WPIC_INITIAL_CONDITION 1\n"
         << run.gamma << ' ' << run.initial_velocity_factor << '\n';
    if (validate) {
        if (!std::filesystem::exists(path)) {
            if (run.initial_velocity_factor != 1.0) {
                throw std::runtime_error("A WPIC checkpoint without an "
                    "initial-condition contract requires --initial-velocity-factor 1.");
            }
            return;
        }
        std::ifstream input(path);
        std::ostringstream previous;
        previous << input.rdbuf();
        if (!input || previous.str() != text.str()) {
            throw std::runtime_error("WPIC restart initial-condition contract differs; "
                "repeat the original --gamma and --initial-velocity-factor.");
        }
        return;
    }
    const auto temporary = directory / "initial_condition_contract.tmp";
    {
        std::ofstream output(temporary);
        output << text.str();
        output.flush();
        if (!output) throw std::runtime_error("Cannot write WPIC initial-condition contract.");
    }
    std::filesystem::rename(temporary, path);
}

}  // namespace

// ---- Self-consistent orbit/collision/output evolution loop -----------------
int ev::apps::run_expanding_weighted_vlasov_pic(int argc, char** argv) {
    // ---- Runtime controls and physical operators -------------------------
    std::signal(SIGUSR1, request_stop);
    std::signal(SIGTERM, request_stop);
    const Options run = parse_options(argc, argv);
    if (run.show_help) {
        print_usage(argv[0]);
        return 0;
    }
    ev::Params p = make_parameters(run);
    if (!run.restart.empty()) {
        validate_restart_radial_mesh(run.restart, p);
        background_contract(run.restart.parent_path(), p, true);
        initial_condition_contract(run.restart.parent_path(), run, true);
    }
    const ev::WeightedPICSubhaloCollisionOptions collision_options =
        make_collision_options(run);
    const bool collisions_enabled = run.subhalo_collision_strength > 0.0;
    ev::WeightedPICFieldGrid grid = ev::make_weighted_pic_field_grid(p, p.Nr);
    const ev::PhaseSpaceCellGeometry compression_geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    PowerLawInitialCondition initial(run.gamma, grid.mesh);

    // ---- Quiet analytic marker sample or versioned continuation ----------
    // Fresh markers sample the requested self-similar distribution.  A
    // restart also restores collision age, evolving clock estimates, and the
    // counter-based random stream so the continuation is deterministic.
    std::vector<ev::WeightedPICMarker> markers;
    ev::WeightedPICCollisionSamplerState collision_sampler{
        run.subhalo_collision_seed, 0};
    double collision_event_age = 0.0;
    double collision_timescale_estimate =
        std::numeric_limits<double>::infinity();
    double potential_timescale_estimate =
        std::numeric_limits<double>::infinity();
    double time = 0.0;
    std::int64_t step = 0;
    if (run.restart.empty()) {
        markers = initialize_markers(run, p, grid, initial);
        if (collisions_enabled) {
            const double specific_internal_energy =
                1.5 * run.subhalo_velocity_scale
                * run.subhalo_velocity_scale;
            for (ev::WeightedPICMarker& marker : markers) {
                marker.internal_energy =
                    marker.mass * specific_internal_energy;
                marker.internal_velocity_dispersion =
                    run.subhalo_velocity_scale;
            }
        }
    } else {
        ev::WeightedPICRestartState state = ev::read_weighted_pic_restart(run.restart);
        markers = std::move(state.markers);
        collision_sampler = state.collision_sampler;
        collision_event_age = state.collision_event_age;
        collision_timescale_estimate =
            state.collision_timescale_estimate;
        potential_timescale_estimate =
            state.potential_timescale_estimate;
        time = state.time;
        step = state.step;
    }
    if (collisions_enabled) {
        for (ev::WeightedPICMarker& marker : markers) {
            if (marker.state != ev::WeightedPICMarkerState::active) {
                continue;
            }
            if (!(marker.internal_velocity_dispersion > 0.0)
                && marker.internal_energy == 0.0) {
                marker.internal_velocity_dispersion =
                    run.subhalo_velocity_scale;
                marker.internal_energy = 1.5 * marker.mass
                    * run.subhalo_velocity_scale
                    * run.subhalo_velocity_scale;
            }
            ev::weighted_pic_subhalo_collision_detail::enforce_internal_state(
                marker, "Expanding WPIC collision initialization");
        }
    }
    if (!(run.end_time > time)) {
        throw std::runtime_error("End time must be after the initial/restart time.");
    }
    if (!run.restart.empty()) {
        ev::require_output_precedes_restart(
            run.output_directory, time,
            {"history.csv", "virial_budget.csv", "collision_history.csv",
             "collision_history_adaptive.csv",
             "collision_history_population_control.csv", "boundary_history.csv"});
    }
    std::filesystem::create_directories(run.output_directory);
    write_radial_mesh_contract_atomic(run.output_directory, p);
    background_contract(run.output_directory, p, false);
    initial_condition_contract(run.output_directory, run, false);
    write_metadata(run, p, initial, markers);

    // ---- Append-safe histories and collision invariant ledgers -----------
    const bool new_history = run.restart.empty()
        || !std::filesystem::exists(run.output_directory / "history.csv");
    std::ofstream history(
        run.output_directory / "history.csv",
        new_history ? std::ios::trunc : std::ios::app);
    if (new_history) {
        history << "step,time,dt,mobile_mass,total_mass,total_energy,max_density,"
                   "minimum_neff,timestep_limit,wall_seconds\n";
    }
    const auto virial_path = run.output_directory / "virial_budget.csv";
    // Budgets refer to the accepted orbit steps of this process segment.
    // The segment start identifies independent checkpoint continuations.
    const auto boundary_path = run.output_directory / "boundary_history.csv";
    std::ofstream boundary_history;
    std::uint64_t segment_wall_impacts = 0;
    double maximum_endpoint_radius = 0.0;
    double radius_maximum_time = time;
    const double boundary_segment_start_time = time;
    const std::int64_t boundary_segment_start_step = step;
    double last_boundary_time = -1.0;
    if (run.boundary_diagnostics) {
        for (const auto& marker : markers) {
            maximum_endpoint_radius = std::max(maximum_endpoint_radius, marker.r);
        }
        const bool exists = std::filesystem::exists(boundary_path);
        boundary_history.open(boundary_path, exists ? std::ios::app : std::ios::trunc);
        if (!boundary_history) throw std::runtime_error("Cannot open boundary_history.csv.");
        if (!exists) boundary_history
            << "step,time,segment_start_step,segment_start_time,"
               "outer_wall_impacts,maximum_accepted_endpoint_radius,"
               "radius_maximum_time,outer_wall_radius\n";
    }
    const auto emit_boundary_history = [&]() {
        if (!run.boundary_diagnostics || time == last_boundary_time) return;
        boundary_history << std::setprecision(17) << step << ',' << time << ','
            << boundary_segment_start_step << ',' << boundary_segment_start_time
            << ',' << segment_wall_impacts << ',' << maximum_endpoint_radius
            << ',' << radius_maximum_time << ',' << p.r_max << '\n';
        boundary_history.flush();
        if (!boundary_history) throw std::runtime_error("Failed writing boundary history.");
        last_boundary_time = time;
    };
    emit_boundary_history();
    const bool new_virial = run.restart.empty()
        || !std::filesystem::exists(virial_path);
    std::ofstream virial_history(virial_path,
        new_virial ? std::ios::trunc : std::ios::app);
    if (!virial_history) throw std::runtime_error("Cannot open virial_budget.csv.");
    if (new_virial) {
        virial_history << "step,time,mobile_mass,resolved_mass,hidden_inner_mass,"
            "I,C,K_pec,K_physical,W_force,C_dot_orbit,I_ddot_half_orbit,"
            "internal_energy,H,H_dot\n";
    }
    const std::filesystem::path legacy_collision_history =
        run.output_directory / "collision_history.csv";
    const std::filesystem::path adaptive_collision_history =
        run.output_directory / "collision_history_adaptive.csv";
    const std::filesystem::path population_collision_history =
        run.output_directory / "collision_history_population_control.csv";
    std::filesystem::path collision_history_path =
        legacy_collision_history;
    if (collisions_enabled
        && !run.restart.empty()
        && std::filesystem::exists(legacy_collision_history)) {
        std::ifstream existing(legacy_collision_history);
        std::string header;
        std::getline(existing, header);
        // Version-5 histories have fewer columns.  Never append version-6
        // clock diagnostics beneath a version-5 header: preserve that file
        // as the fixed-cadence baseline and continue in a clearly named
        // adaptive history.  Later version-6 restarts keep appending there.
        if (header.find("min_collision_time") == std::string::npos) {
            collision_history_path = adaptive_collision_history;
        } else if (
            header.find("repopulation_count") == std::string::npos) {
            collision_history_path = population_collision_history;
        }
    }
    if (collisions_enabled
        && !run.restart.empty()
        && std::filesystem::exists(collision_history_path)) {
        std::ifstream existing(collision_history_path);
        std::string header;
        std::getline(existing, header);
        if (header.find("repopulation_count") == std::string::npos) {
            collision_history_path = population_collision_history;
        }
    }
    const bool new_collision_history = collisions_enabled
        && !std::filesystem::exists(collision_history_path);
    std::ofstream collision_history;
    if (collisions_enabled) {
        collision_history.open(
            collision_history_path,
            new_collision_history ? std::ios::trunc : std::ios::app);
        if (new_collision_history) {
            collision_history
                << "step,time,dt,substeps,trial_pairs,accepted_pairs,"
                   "scattering_events,capture_events,"
                   "inner_cell_visits,inner_scattering_events,"
                   "inner_capture_events,"
                   "represented_scattering_mass,represented_capture_mass,"
                   "orbital_kinetic_energy_change,internal_energy_change,"
                   "total_internal_energy,marker_count,spawned_markers,"
                   "removed_markers,nonlocal_cells,max_dr_over_mfp,"
                   "mfp_pair_samples,min_collision_time,"
                   "min_dynamical_time,min_potential_change_time,"
                   "min_applied_collision_interval,"
                   "max_applied_collision_interval,"
                   "max_interval_over_selection_timescale,"
                   "max_interval_over_retrospective_timescale,"
                   "min_orbital_sigma,max_orbital_sigma,"
                   "max_weight_ratio,max_pair_momentum_error,"
                   "max_pair_total_energy_error,min_internal_sigma,"
                   "max_internal_sigma,min_pair_sigma,max_pair_sigma,"
                   "compression_count,"
                   "compression_markers_before,compression_markers_after,"
                   "compression_groups,compression_retained_groups,"
                   "compression_floor_retained_groups,"
                   "compression_max_moment_residual,"
                   "compression_min_mass_before,compression_max_mass_before,"
                   "compression_min_mass_after,compression_max_mass_after,"
                   "compression_neff_before,compression_neff_after,"
                   "compression_seconds,repopulation_count,"
                   "repopulation_markers_before,repopulation_markers_after,"
                   "repopulation_live_before,repopulation_live_after,"
                   "repopulation_injected_markers,"
                   "repopulation_maximum_factor,"
                   "repopulation_maximum_invariant_residual\n";
        }
    }

    // ---- Initial TSC deposition and expanding Poisson field --------------
    // TSC (triangular-shaped cloud) spreads one marker over neighboring
    // radial cells while preserving represented mass.
    ev::WeightedPICDepositedMoments deposit =
        ev::deposit_weighted_vlasov_pic_parallel(
            markers,
            grid.mesh,
            ev::WeightedPICParticleShape::tsc,
            run.threads,
            true);
    // The field solver integrates the deposited shell masses and unresolved
    // interior mass, then reconstructs the potential and its radial gradient.
    ev::VlasovPoissonField field = ev::solve_weighted_pic_poisson_field(
        deposit, grid.mesh, time, p);
    const double invariant_mobile_mass = ev::weighted_pic_total_marker_mass(markers);
    const double initial_density_l1 = [&] {
        long double error = 0.0L;
        for (std::size_t cell = 0; cell < deposit.shell_mass.size(); ++cell) {
            error += std::abs(
                deposit.shell_mass[cell] - initial.shell_mass(cell));
        }
        return static_cast<double>(error / initial.outer_mass());
    }();
    if (run.restart.empty() && initial_density_l1 > 0.03) {
        throw std::runtime_error(
            "Initial TSC density differs from the analytic IC by more than 3%. ");
    }

    ev::WeightedPICSubhaloCollisionDiagnostics pending_collision;
    struct CompressionWindow {
        std::uint64_t count = 0;
        std::uint64_t markers_before = 0;
        std::uint64_t markers_after = 0;
        std::uint64_t groups = 0;
        std::uint64_t retained_groups = 0;
        std::uint64_t floor_retained_groups = 0;
        double maximum_moment_residual = 0.0;
        double minimum_mass_before =
            std::numeric_limits<double>::infinity();
        double maximum_mass_before = 0.0;
        double minimum_mass_after =
            std::numeric_limits<double>::infinity();
        double maximum_mass_after = 0.0;
        double effective_sample_size_before = 0.0;
        double effective_sample_size_after = 0.0;
        double seconds = 0.0;
    };
    CompressionWindow pending_compression;
    struct PopulationControlWindow {
        std::uint64_t count = 0;
        std::uint64_t markers_before = 0;
        std::uint64_t markers_after = 0;
        std::uint64_t live_before = 0;
        std::uint64_t live_after = 0;
        std::uint64_t injected_markers = 0;
        int maximum_factor = 1;
        double maximum_invariant_residual = 0.0;
    };
    PopulationControlWindow pending_repopulation;
    struct CollisionClockWindow {
        double minimum_dynamical_time =
            std::numeric_limits<double>::infinity();
        double minimum_potential_change_time =
            std::numeric_limits<double>::infinity();
        double minimum_applied_interval =
            std::numeric_limits<double>::infinity();
        double maximum_applied_interval = 0.0;
        double maximum_interval_over_selection_timescale = 0.0;
        double maximum_interval_over_retrospective_timescale = 0.0;
    };
    CollisionClockWindow pending_collision_clocks;
    double last_history_time = -std::numeric_limits<double>::infinity();
    auto emit_history = [&](double dt, double timestep_limit, double wall_seconds) {
        double minimum_neff = std::numeric_limits<double>::infinity();
        for (std::size_t cell = 0; cell < deposit.shell_mass.size(); ++cell) {
            if (deposit.shell_mass[cell] > 1.0e-10 * deposit.deposited_mesh_mass) {
                minimum_neff = std::min(
                    minimum_neff, deposit.effective_marker_number[cell]);
            }
        }
        history << std::setprecision(17)
                << step << ',' << time << ',' << dt << ','
                << deposit.deposited_mesh_mass + deposit.hidden_inner_mass
                << ',' << field.matter_mass << ','
                << total_energy(markers, field, grid, p) << ','
                << *std::max_element(deposit.density.begin(), deposit.density.end())
                << ',' << minimum_neff << ',' << timestep_limit << ','
                << wall_seconds << '\n';
        history.flush();
        const auto virial = marker_virial_budget(markers, field, grid, p);
        const double H = field.background.H;
        const double H_dot = p.vp_universe == "expanding"
            ? -p.vp_alpha / ((time + p.vp_time_offset)
                            * (time + p.vp_time_offset)) : 0.0;
        const long double C_dot = 2.0L * virial.kinetic_peculiar
            + virial.force_virial;
        // For physical positions, half the second derivative of I=sum(m*r^2)
        // is 2*K_phys + W_force + (Hdot+H^2)*I. A ratio near one alone
        // therefore needs the expansion term when interpreting equilibrium.
        const long double I_ddot_half = 2.0L * virial.kinetic_physical
            + virial.force_virial + (H_dot + H * H) * virial.inertia;
        virial_history << std::setprecision(17) << step << ',' << time << ','
            << virial.mass << ',' << deposit.deposited_mesh_mass << ','
            << deposit.hidden_inner_mass << ',' << virial.inertia << ','
            << virial.radial_moment << ',' << virial.kinetic_peculiar << ','
            << virial.kinetic_physical << ',' << virial.force_virial << ','
            << C_dot << ',' << I_ddot_half << ',' << virial.internal_energy
            << ',' << H << ',' << H_dot << '\n';
        virial_history.flush();
        if (!virial_history) throw std::runtime_error("Cannot write virial_budget.csv.");
        if (collisions_enabled) {
            long double internal_energy = 0.0L;
            for (const ev::WeightedPICMarker& marker : markers) {
                internal_energy += marker.internal_energy;
            }
            // CSV uses zero as the portable sentinel for a non-finite
            // diagnostic.  In particular, min_potential_change_time == 0
            // means t_phi = +infinity because no field change rose above the
            // finite-marker noise floor; it is not a vanishing physical time.
            const auto finite_or_zero = [](double value) {
                return std::isfinite(value) ? value : 0.0;
            };
            collision_history << std::setprecision(17)
                << step << ',' << time << ',' << dt << ','
                << pending_collision.substeps << ','
                << pending_collision.trial_pairs << ','
                << pending_collision.accepted_pairs << ','
                << pending_collision.scattering_events << ','
                << pending_collision.capture_events << ','
                << pending_collision.inner_cell_visits << ','
                << pending_collision.inner_scattering_events << ','
                << pending_collision.inner_capture_events << ','
                << pending_collision.represented_scattering_mass << ','
                << pending_collision.represented_capture_mass << ','
                << pending_collision.orbital_kinetic_energy_change << ','
                << pending_collision.internal_energy_change << ','
                << static_cast<double>(internal_energy) << ','
                << markers.size() << ','
                << pending_collision.spawned_markers << ','
                << pending_collision.removed_markers << ','
                << pending_collision.nonlocal_cells << ','
                << pending_collision.maximum_cell_width_over_mean_free_path
                << ',' << pending_collision.mean_free_path_pair_samples
                << ',' << finite_or_zero(
                       pending_collision.minimum_collision_timescale)
                << ',' << finite_or_zero(
                       pending_collision_clocks.minimum_dynamical_time)
                << ',' << finite_or_zero(
                       pending_collision_clocks
                           .minimum_potential_change_time)
                << ',' << finite_or_zero(
                       pending_collision_clocks.minimum_applied_interval)
                << ','
                << pending_collision_clocks.maximum_applied_interval
                << ','
                << pending_collision_clocks
                       .maximum_interval_over_selection_timescale
                << ','
                << pending_collision_clocks
                       .maximum_interval_over_retrospective_timescale
                << ',' << finite_or_zero(
                       pending_collision.minimum_orbital_velocity_dispersion)
                << ','
                << pending_collision.maximum_orbital_velocity_dispersion
                << ',' << pending_collision.maximum_weight_ratio << ','
                << pending_collision.maximum_pair_momentum_error << ','
                << pending_collision.maximum_pair_total_energy_error << ','
                << finite_or_zero(
                       pending_collision.minimum_internal_velocity_dispersion)
                << ','
                << pending_collision.maximum_internal_velocity_dispersion << ','
                << finite_or_zero(
                       pending_collision.minimum_pair_velocity_scale)
                << ','
                << pending_collision.maximum_pair_velocity_scale << ','
                << pending_compression.count << ','
                << pending_compression.markers_before << ','
                << pending_compression.markers_after << ','
                << pending_compression.groups << ','
                << pending_compression.retained_groups << ','
                << pending_compression.floor_retained_groups << ','
                << pending_compression.maximum_moment_residual << ','
                << finite_or_zero(
                       pending_compression.minimum_mass_before) << ','
                << pending_compression.maximum_mass_before << ','
                << finite_or_zero(
                       pending_compression.minimum_mass_after) << ','
                << pending_compression.maximum_mass_after << ','
                << pending_compression.effective_sample_size_before << ','
                << pending_compression.effective_sample_size_after << ','
                << pending_compression.seconds << ','
                << pending_repopulation.count << ','
                << pending_repopulation.markers_before << ','
                << pending_repopulation.markers_after << ','
                << pending_repopulation.live_before << ','
                << pending_repopulation.live_after << ','
                << pending_repopulation.injected_markers << ','
                << pending_repopulation.maximum_factor << ','
                << pending_repopulation.maximum_invariant_residual << '\n';
            collision_history.flush();
            pending_collision = {};
            pending_compression = {};
            pending_repopulation = {};
            pending_collision_clocks = {};
        }
        last_history_time = time;
    };

    // A checkpoint can lie between profile times.  Use the preceding profile
    // index so a resumed run writes the next scheduled profile rather than
    // skipping one. The output preflight has already required every existing
    // profile and history time to be at or before the checkpoint.
    const double output_index_tolerance =
        1.0e-11 * std::max(1.0, std::abs(time));
    std::int64_t output_index = static_cast<std::int64_t>(
        std::floor((time + output_index_tolerance) / run.output_interval));
    if (run.restart.empty()) {
        emit_history(0.0, choose_timestep(deposit, field, grid.mesh, run), 0.0);
        write_profile(run.output_directory, output_index, time, deposit, grid.mesh);
        write_restart_atomic(
            run.output_directory,
            markers,
            time,
            step,
            collision_sampler,
            collision_event_age,
            collision_timescale_estimate,
            potential_timescale_estimate);
        std::cout << std::setprecision(12)
                  << "initialized gamma=" << run.gamma
                  << " markers=" << markers.size()
                  << " mobile_mass=" << invariant_mobile_mass
                  << " initial_core_mass=" << initial.central_mass()
                  << " density_L1=" << initial_density_l1
                  << " low_L_tail_probability="
                  << run.low_angular_tail_probability
                  << " low_L_marker_fraction="
                  << run.low_angular_tail_fraction
                  << " low_L_source_rmax="
                  << run.low_angular_source_rmax
                  << " low_L_source_ppc_multiplier="
                  << run.low_angular_source_ppc_multiplier
                  << " subhalo_collision_strength="
                  << run.subhalo_collision_strength
                  << " subhalo_velocity_scale="
                  << run.subhalo_velocity_scale << '\n';
    }
    if (!run.restart.empty() && run.initial_condition_label != "powerlaw" && time == 0.0) {
        std::ostringstream profile_name;
        profile_name << "profile_" << std::setw(5) << std::setfill('0')
                     << output_index << ".csv";
        if (!std::filesystem::exists(run.output_directory / profile_name.str())) {
            emit_history(0.0, choose_timestep(deposit, field, grid.mesh, run), 0.0);
            write_profile(run.output_directory, output_index, time, deposit, grid.mesh);
        }
    }
    double next_output = (output_index + 1) * run.output_interval;
    if (next_output <= time + 1.0e-12 * std::max(1.0, time)) {
        next_output += run.output_interval;
    }
    const auto wall_start = std::chrono::steady_clock::now();
    ev::WeightedPICSubhaloCollisionWorkspace collision_workspace;
    bool warned_nonlocal_collision_cell = false;
    bool locality_stop_requested = false;
    auto inspect_collision_locality =
        [&](const ev::WeightedPICSubhaloCollisionDiagnostics& stats) {
            const double ratio =
                stats.maximum_cell_width_over_mean_free_path;
            if (run.subhalo_locality_maximum_ratio > 0.0
                && ratio > run.subhalo_locality_maximum_ratio) {
                std::cerr
                    << "LOCALITY STOP: max(Delta r/lambda_mfp)=" << ratio
                    << " exceeds configured limit="
                    << run.subhalo_locality_maximum_ratio
                    << "; checkpointing at t=" << time << ".\n";
                locality_stop_requested = true;
                return;
            }
            if (!warned_nonlocal_collision_cell
                && stats.nonlocal_cells != 0) {
                std::cerr
                    << "WARNING: " << stats.nonlocal_cells
                    << " physical collision cells had Delta r > lambda_mfp; "
                       "continuing with diagnostic max(Delta r/lambda_mfp)="
                    << ratio << '\n';
                warned_nonlocal_collision_cell = true;
            }
        };
    // A version-5 restart (or a fresh run) has no saved t_coll estimate.
    // Evaluate the collision diagnostics with the smallest positive interval
    // before scheduling the first real event.  The expected number of trials
    // underflows safely to zero, so markers and the real random stream remain
    // unchanged; only the evolving-DF pair averages are measured.
    if (collisions_enabled
        && !std::isfinite(collision_timescale_estimate)) {
        ev::WeightedPICCollisionSamplerState diagnostic_sampler =
            collision_sampler;
        const auto diagnostic =
            ev::apply_weighted_pic_subhalo_collision_interval(
                markers,
                grid.mesh,
                std::numeric_limits<double>::min(),
                collision_options,
                diagnostic_sampler,
                collision_workspace,
                run.threads);
        collision_timescale_estimate =
            diagnostic.minimum_collision_timescale;
        inspect_collision_locality(diagnostic);
    }
    bool cadence_stop_requested = false;
    ev::WeightedPICOrbitRollbackBuffer orbit_rollback;
    std::uint64_t rejected_cadence_steps = 0;
    std::uint64_t conservative_cadence_landings = 0;
    // The outer interval controls when to apply the Boltzmann operator.
    // Adaptive cells resolve their own encounter rates inside that interval,
    // while the evolving gravity can still require an earlier global batch.
    auto collision_controlling_timescale = [&] {
        const double dynamical_time =
            minimum_dynamical_timescale(deposit, grid.mesh, p);
        return std::min({
            run.subhalo_adaptive_cell_timestep
                ? std::numeric_limits<double>::infinity()
                : collision_timescale_estimate,
            dynamical_time,
            potential_timescale_estimate,
        });
    };
    auto collision_application_interval = [&] {
        if (!collisions_enabled) {
            return run.subhalo_collision_interval;
        }
        const double dynamical_time =
            minimum_dynamical_timescale(deposit, grid.mesh, p);
        const double selected =
            ev::weighted_pic_collision_selection_interval(
                run.subhalo_collision_interval,
                run.subhalo_collision_timescale_fraction,
                run.subhalo_adaptive_cell_timestep
                    ? std::numeric_limits<double>::infinity()
                    : collision_timescale_estimate,
                dynamical_time,
                potential_timescale_estimate);
        const double representable_interval =
            128.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, std::abs(time));
        if (!(selected > representable_interval)
            || !std::isfinite(selected)) {
            std::cerr
                << "CADENCE STOP: selected collision interval="
                << std::setprecision(17) << selected
                << " is below time-coordinate resolution="
                << representable_interval << " at t=" << time
                << "; checkpointing without sampling a batch.\n";
            cadence_stop_requested = true;
        }
        return std::min(run.subhalo_collision_interval, selected);
    };
    // Record both the timescale used to select the batch and the one measured
    // after it. Their difference shows whether a rapid change occurred during
    // the interval and is distinct from the actual elapsed event age.
    auto record_collision_clocks =
        [&](double applied_interval,
            double selection_controlling_time,
            const ev::WeightedPICSubhaloCollisionDiagnostics& stats) {
            const double dynamical_time =
                minimum_dynamical_timescale(deposit, grid.mesh, p);
            const double controlling_time = std::min({
                stats.minimum_collision_timescale,
                dynamical_time,
                potential_timescale_estimate,
            });
            pending_collision_clocks.minimum_dynamical_time = std::min(
                pending_collision_clocks.minimum_dynamical_time,
                dynamical_time);
            pending_collision_clocks.minimum_potential_change_time = std::min(
                pending_collision_clocks.minimum_potential_change_time,
                potential_timescale_estimate);
            pending_collision_clocks.minimum_applied_interval = std::min(
                pending_collision_clocks.minimum_applied_interval,
                applied_interval);
            pending_collision_clocks.maximum_applied_interval = std::max(
                pending_collision_clocks.maximum_applied_interval,
                applied_interval);
            if (std::isfinite(controlling_time)
                && controlling_time > 0.0) {
                pending_collision_clocks
                    .maximum_interval_over_retrospective_timescale = std::max(
                        pending_collision_clocks
                            .maximum_interval_over_retrospective_timescale,
                        applied_interval / controlling_time);
            }
            if (std::isfinite(selection_controlling_time)
                && selection_controlling_time > 0.0) {
                pending_collision_clocks
                    .maximum_interval_over_selection_timescale = std::max(
                        pending_collision_clocks
                            .maximum_interval_over_selection_timescale,
                        applied_interval / selection_controlling_time);
            }
            collision_timescale_estimate =
                stats.minimum_collision_timescale;
        };
    auto compress_markers_if_needed = [&] {
        if (run.subhalo_compression_trigger_markers == 0
            || markers.size() < run.subhalo_compression_trigger_markers) {
            return;
        }
        const std::size_t markers_before = markers.size();
        long double internal_before = 0.0L;
        for (const ev::WeightedPICMarker& marker : markers) {
            internal_before += marker.internal_energy;
        }
        ev::CanonicalOpenCarrierCompressionOptions options;
        // The production trigger is based on total marker memory.  Once it is
        // crossed, reduce every active marker in the outer mesh or smooth
        // inner-carrier handoff.  The compression constraints preserve the
        // hidden and resolved pieces of that handoff separately; delayed
        // inner and escaped boundary carriers are retained unchanged.
        options.trigger_markers = 1;
        options.minimum_total_markers =
            run.subhalo_compression_minimum_markers;
        options.phase_subcells = run.subhalo_compression_phase_subcells;
        options.threads = run.threads;
        options.preserve_source_cell =
            run.subhalo_compression_preserve_source_cell;
        options.particle_shape = ev::WeightedPICParticleShape::tsc;
        options.minimum_reduction_fraction =
            run.subhalo_compression_minimum_reduction_fraction;
        options.relative_moment_tolerance =
            run.subhalo_compression_tolerance;
        options.minimize_retained_weight_variance =
            run.subhalo_compression_minimize_weight_variance;
        options.maximum_local_mass_amplification =
            run.subhalo_compression_maximum_mass_amplification;
        auto zero_potential = [](double) { return 0.0; };
        const auto diagnostics = ev::compress_canonical_open_carriers(
            markers,
            compression_geometry,
            options,
            zero_potential);
        if (!diagnostics.applied) {
            return;
        }
        long double internal_after = 0.0L;
        for (const ev::WeightedPICMarker& marker : markers) {
            internal_after += marker.internal_energy;
        }
        const double compressed_mass =
            ev::weighted_pic_total_marker_mass(markers);
        const long double internal_scale = std::max({
            1.0e-300L,
            std::abs(internal_before),
            std::abs(internal_after),
        });
        if (std::abs(compressed_mass - invariant_mobile_mass)
                > 256.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, invariant_mobile_mass)
            || std::abs(internal_after - internal_before)
                > run.subhalo_compression_tolerance * internal_scale) {
            throw std::runtime_error(
                "Subhalo marker compression failed invariant closure.");
        }
        ++pending_compression.count;
        pending_compression.markers_before += markers_before;
        pending_compression.markers_after += markers.size();
        pending_compression.groups += diagnostics.groups;
        pending_compression.retained_groups +=
            diagnostics.groups_retained_for_conditioning;
        pending_compression.floor_retained_groups +=
            diagnostics.groups_retained_for_marker_floor;
        pending_compression.maximum_moment_residual = std::max(
            pending_compression.maximum_moment_residual,
            diagnostics.maximum_relative_moment_residual);
        pending_compression.minimum_mass_before = std::min(
            pending_compression.minimum_mass_before,
            diagnostics.minimum_positive_mass_before);
        pending_compression.maximum_mass_before = std::max(
            pending_compression.maximum_mass_before,
            diagnostics.maximum_mass_before);
        pending_compression.minimum_mass_after = std::min(
            pending_compression.minimum_mass_after,
            diagnostics.minimum_positive_mass_after);
        pending_compression.maximum_mass_after = std::max(
            pending_compression.maximum_mass_after,
            diagnostics.maximum_mass_after);
        pending_compression.effective_sample_size_before =
            diagnostics.effective_sample_size_before;
        pending_compression.effective_sample_size_after =
            diagnostics.effective_sample_size_after;
        pending_compression.seconds += diagnostics.elapsed_seconds;
    };
    auto repopulate_markers_if_needed = [&] {
        if (run.subhalo_repopulation_trigger_live_markers == 0) {
            return;
        }
        ev::WeightedPICPopulationControlOptions options;
        options.trigger_live_markers =
            run.subhalo_repopulation_trigger_live_markers;
        options.refinement_factor =
            run.subhalo_repopulation_refinement_factor;
        const auto diagnostics =
            ev::refine_weighted_pic_population_if_needed(markers, options);
        if (!diagnostics.applied) {
            return;
        }
        ++pending_repopulation.count;
        pending_repopulation.markers_before +=
            diagnostics.markers_before;
        pending_repopulation.markers_after += diagnostics.markers_after;
        pending_repopulation.live_before +=
            diagnostics.live_markers_before;
        pending_repopulation.live_after +=
            diagnostics.live_markers_after;
        pending_repopulation.injected_markers +=
            diagnostics.injected_markers;
        pending_repopulation.maximum_factor = std::max(
            pending_repopulation.maximum_factor,
            diagnostics.refinement_factor);
        pending_repopulation.maximum_invariant_residual = std::max(
            pending_repopulation.maximum_invariant_residual,
            diagnostics.maximum_relative_invariant_residual);
        std::cout
            << "population refinement at t=" << std::setprecision(12)
            << time << ": live " << diagnostics.live_markers_before
            << " -> " << diagnostics.live_markers_after
            << ", total " << diagnostics.markers_before
            << " -> " << diagnostics.markers_after
            << ", common factor=" << diagnostics.refinement_factor
            << ", invariant residual="
            << diagnostics.maximum_relative_invariant_residual << '\n';
    };
    auto collision_deadline_is_safe =
        [&](double applied_interval,
            double selected_interval,
            const char* location) {
            if (ev::weighted_pic_collision_deadline_is_roundoff_safe(
                    applied_interval, selected_interval)) {
                return true;
            }
            std::cerr
                << "CADENCE STOP: refusing a collision batch at "
                << location
                << " because accumulated interval="
                << std::setprecision(17) << applied_interval
                << " exceeds the currently known deadline="
                << selected_interval
                << " at t=" << time
                << ".  The pre-collision state will be checkpointed.\n";
            cadence_stop_requested = true;
            return false;
        };
    auto apply_collision_at_current_state =
        [&](double selected_interval,
            double selection_controlling_time,
            const char* location) {
            if (!collision_deadline_is_safe(
                    collision_event_age, selected_interval, location)) {
                return false;
            }
            // Expose the current distribution to collisions for the elapsed
            // event age. Retain the incoming field to measure how encounters
            // changed gravity after the outgoing particles are deposited.
            const double applied_collision_interval = collision_event_age;
            const ev::VlasovPoissonField field_before_collision = field;
            const auto stats =
                ev::apply_weighted_pic_subhalo_collision_interval(
                    markers,
                    grid.mesh,
                    applied_collision_interval,
                    collision_options,
                    collision_sampler,
                    collision_workspace,
                    run.threads);
            ev::weighted_pic_subhalo_collision_detail::accumulate(
                pending_collision, stats);
            collision_event_age = 0.0;
            // A capture batch removes complete partners.  Population control,
            // when enabled, refines the full current live DF only after that
            // physical update has closed its pair invariants.
            repopulate_markers_if_needed();
            compress_markers_if_needed();
            deposit = ev::deposit_weighted_vlasov_pic_parallel(
                markers,
                grid.mesh,
                ev::WeightedPICParticleShape::tsc,
                run.threads,
                true);
            field = ev::solve_weighted_pic_poisson_field(
                deposit, grid.mesh, time, p);
            const double collision_potential_time =
                minimum_potential_change_timescale(
                    field_before_collision,
                    field,
                    applied_collision_interval,
                    deposit);
            potential_timescale_estimate = std::min(
                potential_timescale_estimate,
                collision_potential_time);
            record_collision_clocks(
                applied_collision_interval,
                selection_controlling_time,
                stats);
            // The collision-induced field change starts the next operator
            // interval; orbit changes take a running minimum thereafter.
            potential_timescale_estimate = collision_potential_time;
            inspect_collision_locality(stats);
            return !locality_stop_requested;
        };
    // ---- Coupled Vlasov--Poisson and Boltzmann evolution -----------------
    // The ordinary KDK orbit CFL, the outer collision-operator deadline, and
    // the NTC substeps inside one collision application are distinct clocks.
    // An endpoint field can reveal that t_phi collapsed inside a proposed KDK
    // step.  Adaptive runs therefore treat KDK as a transaction: save (r,w),
    // probe the endpoint clocks, and restore/retry before committing if the
    // full collision age would cross the newly known operator deadline.
    double previous_collision_decision_time = time;
    while (time < run.end_time
           && !stop_requested.load(std::memory_order_relaxed)
           && !locality_stop_requested
           && !cadence_stop_requested) {
        const double timestep_limit = choose_timestep(deposit, field, grid.mesh, run);
        // Land on the next physical event, requested output or final time
        // without stepping past it. Output spacing also bounds this interval.
        double dt = std::min({
            timestep_limit,
            run.end_time - time,
            std::max(0.0, next_output - time),
        });
        double selected_collision_interval =
            run.subhalo_collision_interval;
        if (collisions_enabled) {
            selected_collision_interval =
                collision_application_interval();
            if (cadence_stop_requested) {
                break;
            }
            // An event can be due within floating-point time resolution.
            // The optional assessment decides whether to apply its accumulated
            // exposure now, advance safely, or stop with the incoming state.
            if (run.collision_prestep_coordinate_due) {
                const auto due = ev::assess_weighted_pic_pre_step_collision_cadence(
                    collision_event_age, selected_collision_interval, time,
                    previous_collision_decision_time);
                previous_collision_decision_time = time;
                if (due.status != ev::WeightedPICPreStepCollisionStatus::advance) {
                    std::cerr << "CADENCE PRE-STEP: t=" << std::setprecision(17)
                        << time << " age=" << collision_event_age
                        << " deadline=" << selected_collision_interval
                        << " remaining=" << due.remaining_interval
                        << " coordinate_resolution=" << due.coordinate_resolution
                        << " action=" << ev::weighted_pic_pre_step_collision_status_name(due.status)
                        << '\n';
                    if (due.status != ev::WeightedPICPreStepCollisionStatus::apply_collision_now) {
                        cadence_stop_requested = true;
                        break;
                    }
                    if (!apply_collision_at_current_state(
                            selected_collision_interval,
                            collision_controlling_timescale(),
                            "a coordinate-resolution-limited pre-step deadline")) {
                        break;
                    }
                    continue;
                }
            }
            dt = std::min(
                dt,
                std::max(
                    0.0,
                    selected_collision_interval - collision_event_age));
        }
        if (!(dt > 0.0)) {
            if (collisions_enabled
                && collision_event_age
                    >= selected_collision_interval
                        - 1.0e-12
                            * std::max(
                                1.0, selected_collision_interval)) {
                // A physical clock shortened beneath the already accumulated
                // event age.  A roundoff-close age may be applied now; a
                // genuine overshoot is never sampled retroactively.
                if (!apply_collision_at_current_state(
                        selected_collision_interval,
                        collision_controlling_timescale(),
                        "the pre-step deadline")) {
                    break;
                }
                continue;
            }
            next_output += run.output_interval;
            continue;
        }
        // Kick--drift--kick (KDK) advances collisionless characteristics
        // using the old and endpoint self-consistent gravitational fields.
        // Only an adaptive collisional run needs rollback storage: fixed
        // cadence and collisionless trajectories have no endpoint-dependent
        // operator deadline.
        const bool transactional_cadence =
            collisions_enabled
            && run.subhalo_collision_timescale_fraction > 0.0;
        if (transactional_cadence) {
            ev::capture_weighted_pic_orbit_rollback(
                markers, orbit_rollback, run.threads);
        }
        ev::WeightedPICDepositedMoments endpoint_deposit;
        ev::WeightedPICOrbitDiagnostics orbit_step_diagnostics;
        ev::VlasovPoissonField endpoint_field;
        int cadence_retry_depth = 0;
        bool apply_collision_before_step = false;
        bool apply_collision_after_retried_step = false;
        double early_collision_deadline = 0.0;
        double early_collision_controlling_time =
            std::numeric_limits<double>::infinity();
        while (true) {
            // First kick with the old force and drift to the trial endpoint.
            // Re-deposit there before the closing kick so gravity responds
            // to the particles at the appropriate KDK stage.
            ev::begin_weighted_pic_self_consistent_kdk_parallel(
                markers,
                time,
                dt,
                grid.radial_coordinate,
                grid.radius,
                field,
                p,
                run.threads,
                run.boundary_diagnostics ? &orbit_step_diagnostics : nullptr);
            endpoint_deposit =
                ev::deposit_weighted_vlasov_pic_parallel(
                    markers,
                    grid.mesh,
                    ev::WeightedPICParticleShape::tsc,
                    run.threads,
                    true);
            endpoint_field =
                ev::solve_weighted_pic_poisson_field(
                    endpoint_deposit, grid.mesh, time + dt, p);
            const double endpoint_potential_time =
                minimum_potential_change_timescale(
                    field, endpoint_field, dt, endpoint_deposit);
            const double candidate_potential_time = std::min(
                potential_timescale_estimate,
                endpoint_potential_time);

            if (transactional_cadence) {
                const double endpoint_dynamical_time =
                    minimum_dynamical_timescale(
                        endpoint_deposit, grid.mesh, p);
                const auto cadence =
                    ev::assess_weighted_pic_endpoint_collision_cadence(
                        collision_event_age,
                        dt,
                        time,
                        run.subhalo_collision_interval,
                        run.subhalo_collision_timescale_fraction,
                        run.subhalo_adaptive_cell_timestep
                            ? std::numeric_limits<double>::infinity()
                            : collision_timescale_estimate,
                        endpoint_dynamical_time,
                        candidate_potential_time);
                if (!cadence.accept_step) {
                    ev::restore_weighted_pic_orbit_rollback(
                        markers, orbit_rollback, run.threads);
                    ++cadence_retry_depth;
                    ++rejected_cadence_steps;
                    if (cadence.apply_collision_now) {
                        apply_collision_before_step = true;
                        early_collision_deadline =
                            cadence.selected_interval;
                        early_collision_controlling_time =
                            cadence.controlling_timescale;
                        std::cerr
                            << "CADENCE EARLY APPLY: t="
                            << std::setprecision(17) << time
                            << " age=" << collision_event_age
                            << " deadline=" << cadence.selected_interval
                            << " lies within coordinate resolution; "
                               "applying at the rolled-back pre-step state.\n";
                        break;
                    }
                    if (cadence_retry_depth == 1
                        || cadence_retry_depth % 8 == 0) {
                        std::cerr
                            << "CADENCE RETRY: t="
                            << std::setprecision(17) << time
                            << " age=" << collision_event_age
                            << " proposed_dt=" << dt
                            << " endpoint_deadline="
                            << cadence.selected_interval
                            << " endpoint_ratio="
                            << cadence
                                   .interval_over_controlling_timescale
                            << " retry_dt=" << cadence.retry_interval
                            << '\n';
                    }
                    if (cadence.unresolved_overdue
                        || cadence_retry_depth >= 64) {
                        std::cerr
                            << "CADENCE STOP: the endpoint clock remained "
                               "overdue after "
                            << cadence_retry_depth
                            << " rollback probes at t=" << time
                            << "; accumulated collision age="
                            << collision_event_age
                            << ", last endpoint deadline="
                            << cadence.selected_interval
                            << ".  Refusing to apply a temporally "
                               "misordered collision batch.\n";
                        cadence_stop_requested = true;
                        break;
                    }
                    dt = cadence.retry_interval;
                    continue;
                }
                if (cadence_retry_depth > 0) {
                    // The retried endpoint is on the safe side of the
                    // endpoint-measured collision deadline.  Apply the
                    // collision there after committing KDK.  Waiting for the
                    // exact equality would start a new outer step and repeat
                    // the rollback/halve cycle, producing many tiny orbit
                    // steps without resolving any additional physics.
                    //
                    // This is conservative: the applied interval is shorter
                    // than epsilon_op min(t_coll,t_dyn,t_phi), so it cannot
                    // weaken the temporal-resolution bound.  The collision
                    // clock is reset only after the accepted orbit endpoint
                    // has been committed, preserving Vlasov-then-Boltzmann
                    // operator order.
                    apply_collision_after_retried_step = true;
                    ++conservative_cadence_landings;
                }
            }

            ev::finish_weighted_pic_self_consistent_kdk_parallel(
                markers,
                time + dt,
                dt,
                grid.radial_coordinate,
                grid.radius,
                endpoint_field,
                p,
                run.threads);
            potential_timescale_estimate =
                candidate_potential_time;
            break;
        }
        if (apply_collision_before_step) {
            if (!apply_collision_at_current_state(
                    early_collision_deadline,
                    early_collision_controlling_time,
                    "a coordinate-resolution-limited pre-step boundary")) {
                break;
            }
            continue;
        }
        if (cadence_stop_requested) {
            break;
        }
        time += dt;
        // Collision boundaries introduce additional floating-point additions.
        // Snap a roundoff-close endpoint before output scheduling so the final
        // profile is written once, at the requested time, rather than once
        // just below it and again after a tiny cleanup step.
        if (time >= run.end_time
                - 1.0e-12 * std::max(1.0, run.end_time)
            && time <= run.end_time
                + 1.0e-12 * std::max(1.0, run.end_time)) {
            time = run.end_time;
        }
        ++step;
        if (run.boundary_diagnostics) {
            segment_wall_impacts += orbit_step_diagnostics.reflecting_wall_impacts;
            if (orbit_step_diagnostics.maximum_endpoint_radius > maximum_endpoint_radius) {
                maximum_endpoint_radius = orbit_step_diagnostics.maximum_endpoint_radius;
                radius_maximum_time = time;
            }
        }
        collision_event_age += dt;
        deposit = std::move(endpoint_deposit);
        field = std::move(endpoint_field);

        const bool final = time >= run.end_time
            - 1.0e-12 * std::max(1.0, run.end_time);
        selected_collision_interval = collisions_enabled
            ? collision_application_interval()
            : run.subhalo_collision_interval;
        if (cadence_stop_requested) {
            break;
        }
        const bool collision_due = collisions_enabled
            && (apply_collision_after_retried_step
                || collision_event_age
                    >= selected_collision_interval
                        - 1.0e-12
                            * std::max(
                                1.0, selected_collision_interval)
                || final);
        if (collision_due) {
            // Sample unequal-mass subhalo scattering and capture, then
            // redeposit because capture changes marker masses and count.
            if (!apply_collision_at_current_state(
                    selected_collision_interval,
                    collision_controlling_timescale(),
                    "the accepted KDK endpoint")) {
                break;
            }
        }

        const bool output_due = time >= next_output
            - 1.0e-11 * std::max(1.0, next_output);
        const double wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();
        // ---- Cadenced diagnostics, profiles, and atomic checkpoints -------
        if (step % run.history_every == 0 || output_due || final) {
            if (output_due || final) {
                deposit = ev::deposit_weighted_vlasov_pic_parallel(
                    markers,
                    grid.mesh,
                    ev::WeightedPICParticleShape::tsc,
                    run.threads,
                    true);
            }
            emit_history(dt, timestep_limit, wall_seconds);
        }
        if (output_due || final) {
            ++output_index;
            write_profile(run.output_directory, output_index, time, deposit, grid.mesh);
            next_output += run.output_interval;
            emit_boundary_history();
        }
        // Profiles are small and may be frequent for smooth animations;
        // checkpoints contain every marker and are intentionally less frequent.
        // The unconditional write after the loop still checkpoints immediately
        // on a scheduler signal.
        if (step % run.restart_every == 0 || final) {
            emit_boundary_history();
            write_restart_atomic(
                run.output_directory,
                markers,
                time,
                step,
                collision_sampler,
                collision_event_age,
                collision_timescale_estimate,
                potential_timescale_estimate);
        }
        if (step % run.history_every == 0 || output_due || final) {
            std::cout << std::setprecision(10)
                      << "gamma=" << run.gamma << " step=" << step
                      << " t=" << time << " dt=" << dt
                      << " dt_limit=" << timestep_limit
                      << " mass="
                      << deposit.deposited_mesh_mass + deposit.hidden_inner_mass
                      << " wall=" << wall_seconds << '\n';
        }
    }
    if ((locality_stop_requested || cadence_stop_requested)
        && std::abs(last_history_time - time)
            > 1.0e-12 * std::max(1.0, std::abs(time))) {
        const double wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();
        emit_history(
            0.0,
            choose_timestep(deposit, field, grid.mesh, run),
            wall_seconds);
    }
    emit_boundary_history();
    // ---- Final conservation/locality disposition -------------------------
    write_restart_atomic(
        run.output_directory,
        markers,
        time,
        step,
        collision_sampler,
        collision_event_age,
        collision_timescale_estimate,
        potential_timescale_estimate);
    const double final_mass = ev::weighted_pic_total_marker_mass(markers);
    if (std::abs(final_mass - invariant_mobile_mass)
        // Compression changes both the marker count and summation order.
        // Use the same roundoff envelope enforced immediately after each
        // compression/collision batch; tighter here can reject an evolution
        // that already passed every physical conservation check.
        > 256.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, invariant_mobile_mass)) {
        throw std::runtime_error("Expanding WPIC changed invariant marker mass.");
    }
    // A strict locality or cadence exit carries a diagnostic state.  Its
    // restartability requires review of the failed condition and the last
    // accepted state, separately from a scheduler-only checkpoint exit.
    if (locality_stop_requested) {
        std::cout << "checkpointed at the configured mean-free-path locality "
                     "limit at t=" << time << '\n';
        return 98;
    }
    if (cadence_stop_requested) {
        std::cout
            << "checkpointed before a temporally unresolved collision "
               "deadline at t="
            << time << " after " << rejected_cadence_steps
            << " rejected KDK probes\n";
        return 97;
    }
    if (stop_requested.load(std::memory_order_relaxed) && time < run.end_time) {
        std::cout << "checkpointed after scheduler signal at t=" << time << '\n';
        return 99;
    }
    std::cout
        << "expanding weighted PIC run completed at t=" << time
        << " after " << rejected_cadence_steps
        << " rejected KDK probes and "
        << conservative_cadence_landings
        << " conservative post-retry collision landings\n";
    return 0;
}
