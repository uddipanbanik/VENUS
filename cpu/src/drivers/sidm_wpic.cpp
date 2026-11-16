// Evolve a static, spherical halo of elastically interacting dark matter.
// The input is a complete weighted-PIC restart or a spherical particle file.
// Particle mass, radius r, radial velocity w and eta=L^2/2 determine its
// contribution to density and its Newtonian orbit. The selected SIDM law
// supplies the scattering cross section as a function of relative speed.
//
// Gravity and collisions have separate update times. Between field updates,
// advance the particle trajectories through the frozen Poisson potential.
// When an encounter batch is due, group particles by their current radii,
// sample elastic encounters, and retain the outgoing velocities. Reconstruct
// the radial moments and refresh gravity when the field clock requires it.
// The optional phase-space grid is a second representation of the particles.
// A particle remains a continuous "carrier" when projecting its mass onto
// that grid would exceed the allowed local moment error.
//
// The outputs are radial profiles, core and energy histories, collision and
// reconstruction diagnostics, and checkpoints. Checkpoints retain the frozen
// field, random counters and the ages of both event clocks. Together these
// quantities identify where to resume between field and collision updates.
// Elastic scattering preserves each marker mass. Subhalo capture has its own
// implementation in the subhalo driver.

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "ev/config.hpp"
#include "ev/restart_output_guard.hpp"
#include "ev/spherical_dsmc_solver.hpp"
#include "ev/wpic_task_profile.hpp"
#include "ev/apps/wpic_driver_entry.hpp"
#include "ev/weighted_pic_boltzmann_event.hpp"
#include "ev/weighted_pic_frank_restart_import.hpp"

namespace {

volatile std::sig_atomic_t stop_requested = 0;

void request_orderly_stop(int) {
    stop_requested = 1;
}

// Physical controls select the material cross section and gravity model.
// Shell populations and quadrature set numerical sampling, while the event
// intervals determine when the frozen field and encounter state are updated.
struct RunOptions {
    // Exactly one restart source is required.  A Frank/SphDSMC restart
    // contains spherical particles and may be unequal mass; a PIC restart
    // preserves the complete weighted-marker state and both deterministic
    // random-number streams.
    std::filesystem::path frank_restart;
    std::filesystem::path pic_restart;
    // Explicitly start a new field interval when changing the mesh or other
    // controls. Ordinary v7 continuation restores the saved frozen field.
    bool refresh_restart_field = false;
    std::filesystem::path output_directory = "weighted_pic_sidm_output";
    double end_time = 0.02;
    // Optional equal-mass import subsampling reduces marker count for short
    // runs.  A full import is mandatory for unequal-mass sources and for
    // quantitative rho_core/profile comparisons.
    std::size_t marker_limit = 0;
    // Maximum interval between inexpensive clock probes.  Physical field and
    // collision deadlines can always shorten this interval.
    double maximum_timestep = 0.02;
    // Fractions of the local dynamical, potential-change, and collision
    // timescales used by the independent event clocks.  These are accuracy
    // controls: they are not cross sections or physical coupling constants.
    double dynamical_eta = 0.01;
    double potential_event_eta = 0.1;
    double collision_event_eta = 0.1;
    // The field mesh resolves density and potential in radius.  The w and eta
    // axes resolve radial velocity and eta=L^2/2 during conservative remaps.
    int radial_cells = 256;
    int radial_velocity_cells = 32;
    int eta_cells = 96;
    double radial_minimum = 1.0e-3;
    double radial_maximum = 1.0e6;
    double radial_velocity_maximum = 5.0;
    double angular_momentum_maximum = 400.0;
    double eta_grid_power = 8.0;
    int remap_candidate_radius = 8;
    // Hard fraction of active mass allowed to remain as explicitly tracked
    // continuous remap carriers.  Preflight the value on the representative
    // continuous unequal-mass state; it is not a physics-error tolerance.
    double maximum_carrier_mass_fraction = 0.01;
    // Persistent markers are the evolved kinetic state.  The full (r,w,eta)
    // projection is useful for phase-space output, but need not be repeated at
    // every physical event when only radial moments and marker checkpoints are
    // written.  By default it is synchronized once at normal final output;
    // the explicit skip-final diagnostic flag can suppress that projection.
    bool phase_space_projection_at_events = true;
    bool phase_space_projection_at_final_output = true;
    int output_every = 1;
    int restart_every = 50;
    unsigned int threads = 1;
    ev::WeightedPICParticleShape shape = ev::WeightedPICParticleShape::tsc;
    // sigma0 fixes the normalization of Sigma(u); u0 and sidm_power set the
    // velocity scale and exponent for the power-law model.  The Yukawa
    // parameters set a numerical speed scale mphi/mchi and amplitude
    // 4*pi*alpha^2*mchi^2/mphi^4, interpreted here as cross section per mass.
    // Calibrate that amplitude after converting a physical microscopic model
    // to simulation units; alpha is then an effective input amplitude.
    std::string sidm_cross_section = "constant";
    double collision_strength = 1.0;
    double sidm_sigma0 = 32.354198286197665;
    double sidm_u0 = 1.0;
    double sidm_power = 0.0;
    double yukawa_alpha = 1.0;
    double yukawa_mchi = 1.0;
    double yukawa_mphi = 1.0;
    // Collision shells are adaptive radial neighborhoods.  target controls
    // statistical sampling, minimum rejects undersampled shells, and the
    // collision CFL limits the majorant trial-rate exposure per substep.
    // Several trials, and therefore several accepted encounters, can occur in
    // one substep when repeat encounters are enabled.
    std::size_t collision_target_shell_markers = 4096;
    std::size_t collision_minimum_shell_markers = 16;
    double collision_cfl = 1.0;
    int collision_maximum_substeps = 200;
    // When enabled, every disjoint radial shell advances its own collision
    // clock.  The outer operator batch remains synchronized with gravity, but
    // slow shells are no longer revisited at the dense core's timestep.
    bool collision_adaptive_shell_timestep = false;
    double collision_adaptive_maximum_interval = 0.02;
    bool fail_on_nonlocal_collision_shell = true;
};

// ---- Runtime and collision-model options -----------------------------------
void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0
        << " (--frank-restart PATH | --pic-restart PATH) [options]\n"
#if defined(VENUS_CPU_SIDM_WPIC_CONSTANT_PROFILE)
        << "Locked task profile: constant-cross-section SIDM WPIC; "
           "--sidm-cross-section may only be constant and collision strength "
           "must be positive.\n"
#elif defined(VENUS_CPU_SIDM_WPIC_VELOCITY_DEPENDENT_PROFILE)
        << "Locked task profile: velocity-dependent SIDM WPIC; "
           "--sidm-cross-section must be powerlaw or yukawa_born and collision "
           "strength must be positive.\n"
#else
        << "Compatibility driver: constant, power-law, and Yukawa/Born SIDM "
           "laws are selectable.\n"
#endif
        << "  --output-directory DIR --end-time X --marker-limit N\n"
        << "  --refresh-restart-field (rebuild field after changing restart controls)\n"
        << "  --maximum-timestep X --dynamical-eta X\n"
        << "  --potential-event-eta X --collision-event-eta X\n"
        << "  --radial-cells N --radial-velocity-cells N --eta-cells N\n"
        << "  --radial-minimum X --radial-maximum X\n"
        << "  --radial-velocity-maximum X --angular-momentum-maximum X\n"
        << "  --eta-grid-power X --remap-candidate-radius N\n"
        << "  --maximum-carrier-mass-fraction X\n"
        << "  --phase-space-projection-at-output-only\n"
        << "  --skip-final-phase-space-projection\n"
        << "  --output-every N --restart-every N --threads N --shape cic|tsc\n"
#if defined(VENUS_CPU_SIDM_WPIC_CONSTANT_PROFILE)
        << "  --sidm-cross-section constant\n"
#elif defined(VENUS_CPU_SIDM_WPIC_VELOCITY_DEPENDENT_PROFILE)
        << "  --sidm-cross-section powerlaw|yukawa_born\n"
#else
        << "  --sidm-cross-section constant|powerlaw|yukawa_born\n"
#endif
        << "  --collision-strength X --sidm-sigma0 X --sidm-u0 X --sidm-power X\n"
        << "  --yukawa-alpha X --yukawa-mchi X --yukawa-mphi X\n"
        << "  --collision-target-shell-markers N\n"
        << "  --collision-minimum-shell-markers N --collision-cfl X\n"
        << "  --collision-maximum-substeps N\n"
        << "  --collision-adaptive-shell-timestep\n"
        << "  --collision-adaptive-maximum-interval X\n"
        << "  --allow-nonlocal-collision-shell\n"
        << "  --help, -h\n";
}

RunOptions parse_options(int argc, char** argv) {
    RunOptions result;
    auto value = [&](int& argument, const char* name) {
        if (++argument >= argc) {
            throw std::runtime_error(std::string("Missing value for ") + name);
        }
        return std::string(argv[argument]);
    };
    for (int argument = 1; argument < argc; ++argument) {
        const std::string name = argv[argument];
        if (name == "--help" || name == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (name == "--frank-restart") {
            result.frank_restart = value(argument, "--frank-restart");
        } else if (name == "--pic-restart") {
            result.pic_restart = value(argument, "--pic-restart");
        } else if (name == "--refresh-restart-field") {
            result.refresh_restart_field = true;
        } else if (name == "--output-directory") {
            result.output_directory = value(argument, "--output-directory");
        } else if (name == "--end-time") {
            result.end_time = std::stod(value(argument, "--end-time"));
        } else if (name == "--marker-limit") {
            result.marker_limit = static_cast<std::size_t>(
                std::stoull(value(argument, "--marker-limit")));
        } else if (name == "--maximum-timestep") {
            result.maximum_timestep =
                std::stod(value(argument, "--maximum-timestep"));
        } else if (name == "--dynamical-eta") {
            result.dynamical_eta = std::stod(value(argument, "--dynamical-eta"));
        } else if (name == "--potential-event-eta") {
            result.potential_event_eta =
                std::stod(value(argument, "--potential-event-eta"));
        } else if (name == "--collision-event-eta") {
            result.collision_event_eta =
                std::stod(value(argument, "--collision-event-eta"));
        } else if (name == "--radial-cells") {
            result.radial_cells = std::stoi(value(argument, "--radial-cells"));
        } else if (name == "--radial-velocity-cells") {
            result.radial_velocity_cells =
                std::stoi(value(argument, "--radial-velocity-cells"));
        } else if (name == "--eta-cells") {
            result.eta_cells = std::stoi(value(argument, "--eta-cells"));
        } else if (name == "--radial-minimum") {
            result.radial_minimum =
                std::stod(value(argument, "--radial-minimum"));
        } else if (name == "--radial-maximum") {
            result.radial_maximum =
                std::stod(value(argument, "--radial-maximum"));
        } else if (name == "--radial-velocity-maximum") {
            result.radial_velocity_maximum =
                std::stod(value(argument, "--radial-velocity-maximum"));
        } else if (name == "--angular-momentum-maximum") {
            result.angular_momentum_maximum =
                std::stod(value(argument, "--angular-momentum-maximum"));
        } else if (name == "--eta-grid-power") {
            result.eta_grid_power =
                std::stod(value(argument, "--eta-grid-power"));
        } else if (name == "--remap-candidate-radius") {
            result.remap_candidate_radius =
                std::stoi(value(argument, "--remap-candidate-radius"));
        } else if (name == "--maximum-carrier-mass-fraction") {
            result.maximum_carrier_mass_fraction = std::stod(
                value(argument, "--maximum-carrier-mass-fraction"));
        } else if (name == "--phase-space-projection-at-output-only") {
            result.phase_space_projection_at_events = false;
        } else if (name == "--skip-final-phase-space-projection") {
            result.phase_space_projection_at_final_output = false;
        } else if (name == "--output-every") {
            result.output_every = std::stoi(value(argument, "--output-every"));
        } else if (name == "--restart-every") {
            result.restart_every = std::stoi(value(argument, "--restart-every"));
        } else if (name == "--threads") {
            result.threads = static_cast<unsigned int>(
                std::stoul(value(argument, "--threads")));
        } else if (name == "--shape") {
            const std::string shape = value(argument, "--shape");
            if (shape == "cic") {
                result.shape = ev::WeightedPICParticleShape::cic;
            } else if (shape == "tsc") {
                result.shape = ev::WeightedPICParticleShape::tsc;
            } else {
                throw std::runtime_error("--shape must be cic or tsc.");
            }
        } else if (name == "--sidm-cross-section") {
            result.sidm_cross_section = value(argument, "--sidm-cross-section");
        } else if (name == "--collision-strength") {
            result.collision_strength =
                std::stod(value(argument, "--collision-strength"));
        } else if (name == "--sidm-sigma0") {
            result.sidm_sigma0 = std::stod(value(argument, "--sidm-sigma0"));
        } else if (name == "--sidm-u0") {
            result.sidm_u0 = std::stod(value(argument, "--sidm-u0"));
        } else if (name == "--sidm-power") {
            result.sidm_power = std::stod(value(argument, "--sidm-power"));
        } else if (name == "--yukawa-alpha") {
            result.yukawa_alpha = std::stod(value(argument, "--yukawa-alpha"));
        } else if (name == "--yukawa-mchi") {
            result.yukawa_mchi = std::stod(value(argument, "--yukawa-mchi"));
        } else if (name == "--yukawa-mphi") {
            result.yukawa_mphi = std::stod(value(argument, "--yukawa-mphi"));
        } else if (name == "--collision-target-shell-markers") {
            result.collision_target_shell_markers = static_cast<std::size_t>(
                std::stoull(value(argument, "--collision-target-shell-markers")));
        } else if (name == "--collision-minimum-shell-markers") {
            result.collision_minimum_shell_markers = static_cast<std::size_t>(
                std::stoull(value(argument, "--collision-minimum-shell-markers")));
        } else if (name == "--collision-cfl") {
            result.collision_cfl = std::stod(value(argument, "--collision-cfl"));
        } else if (name == "--collision-maximum-substeps") {
            result.collision_maximum_substeps =
                std::stoi(value(argument, "--collision-maximum-substeps"));
        } else if (name == "--collision-adaptive-shell-timestep") {
            result.collision_adaptive_shell_timestep = true;
        } else if (name == "--collision-adaptive-maximum-interval") {
            result.collision_adaptive_maximum_interval = std::stod(
                value(argument, "--collision-adaptive-maximum-interval"));
        } else if (name == "--allow-nonlocal-collision-shell") {
            result.fail_on_nonlocal_collision_shell = false;
        } else {
            throw std::runtime_error("Unknown static SIDM WPIC option: " + name);
        }
    }
    if (result.frank_restart.empty() == result.pic_restart.empty()) {
        throw std::runtime_error(
            "Specify exactly one of --frank-restart or --pic-restart.");
    }
    if (!result.pic_restart.empty() && result.marker_limit != 0) {
        throw std::runtime_error(
            "--marker-limit applies only when importing a Frank restart.");
    }
    if (result.refresh_restart_field && result.pic_restart.empty()) {
        throw std::runtime_error("--refresh-restart-field requires --pic-restart.");
    }
#if defined(VENUS_CPU_SIDM_WPIC_CONSTANT_PROFILE)
    try {
        ev::validate_sidm_wpic_task_profile(
            ev::SIDMWPICTaskProfile::constant_cross_section,
            result.sidm_cross_section,
            result.collision_strength);
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        std::exit(1);
    }
#elif defined(VENUS_CPU_SIDM_WPIC_VELOCITY_DEPENDENT_PROFILE)
    try {
        ev::validate_sidm_wpic_task_profile(
            ev::SIDMWPICTaskProfile::velocity_dependent_cross_section,
            result.sidm_cross_section,
            result.collision_strength);
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        std::exit(1);
    }
#endif
    if (!(result.end_time > 0.0) || !(result.maximum_timestep > 0.0)
        || !(result.dynamical_eta > 0.0)
        || !(result.potential_event_eta > 0.0)
        || !(result.potential_event_eta <= 1.0)
        || !std::isfinite(result.potential_event_eta)
        || !(result.collision_event_eta > 0.0)
        || !(result.collision_event_eta <= 1.0)
        || !std::isfinite(result.collision_event_eta)
        || result.radial_cells < 16
        || result.radial_velocity_cells < 4
        || result.eta_cells < 2
        || !(result.radial_minimum > 0.0)
        || !(result.radial_maximum > result.radial_minimum)
        || !(result.radial_velocity_maximum > 0.0)
        || !std::isfinite(result.radial_velocity_maximum)
        || !(result.angular_momentum_maximum > 0.0)
        || !std::isfinite(result.angular_momentum_maximum)
        || !(result.eta_grid_power >= 1.0)
        || !std::isfinite(result.eta_grid_power)
        || result.remap_candidate_radius < 1
        || !(result.maximum_carrier_mass_fraction >= 0.0)
        || !(result.maximum_carrier_mass_fraction < 1.0)
        || !std::isfinite(result.maximum_carrier_mass_fraction)
        || result.output_every < 1 || result.restart_every < 1
        || result.threads == 0
        || !(result.sidm_cross_section == "constant"
             || result.sidm_cross_section == "powerlaw"
             || result.sidm_cross_section == "yukawa_born")
        || !(result.collision_strength >= 0.0)
        || !std::isfinite(result.collision_strength)
        || !(result.sidm_sigma0 >= 0.0) || !std::isfinite(result.sidm_sigma0)
        || !(result.sidm_u0 >= 0.0) || !std::isfinite(result.sidm_u0)
        || !(result.sidm_power >= 0.0) || !std::isfinite(result.sidm_power)
        || !(result.yukawa_alpha > 0.0) || !std::isfinite(result.yukawa_alpha)
        || !(result.yukawa_mchi > 0.0) || !std::isfinite(result.yukawa_mchi)
        || !(result.yukawa_mphi > 0.0) || !std::isfinite(result.yukawa_mphi)
        || result.collision_target_shell_markers < 2
        || result.collision_minimum_shell_markers < 2
        || result.collision_minimum_shell_markers
            > result.collision_target_shell_markers
        || !(result.collision_cfl > 0.0)
        || !std::isfinite(result.collision_cfl)
        || result.collision_maximum_substeps < 1
        || !(result.collision_adaptive_maximum_interval > 0.0)
        || !std::isfinite(result.collision_adaptive_maximum_interval)) {
        throw std::runtime_error("Invalid static SIDM WPIC options.");
    }
    if (result.sidm_cross_section == "powerlaw"
        && !(result.sidm_u0 > 0.0)) {
        throw std::runtime_error("Power-law SIDM requires --sidm-u0 > 0.");
    }
    if (result.collision_adaptive_shell_timestep
        && (result.sidm_cross_section != "constant"
            || result.collision_cfl > 1.0)) {
        throw std::runtime_error(
            "CPU adaptive SIDM shell clocks require constant cross section "
            "and --collision-cfl <= 1; every occupied shell must also have "
            "equal marker masses (checked when the restart is loaded).");
    }
    return result;
}

ev::Params make_parameters(const RunOptions& run) {
    // Translate this task-specific interface into the common VENUS
    // parameter record.  The orbit solve is collisionless Vlasov--Poisson;
    // the separate Boltzmann event inserts local elastic SIDM scattering.
    ev::Params p;
    p.physics_model = "vlasov_poisson";
    p.vp_universe = "static";
    p.vp_gravity = 1.0;
    p.vp_poisson_source = "total_density";
    p.vp_interpolation = "hamiltonian_sldg";
    p.vp_orbit_regularization = "auto";
    p.vp_orbit_energy_tolerance = 1.0e-8;
    p.vp_orbit_lc_relative_tolerance = 1.0e-9;
    p.vp_orbit_lc_absolute_tolerance = 1.0e-12;
    p.vp_orbit_lc_max_steps = 20000;
    p.vp_orbit_lc_max_refinements = 3;
    p.vp_dynamical_eta = run.dynamical_eta;
    p.vp_potential_variation_eta = run.potential_event_eta;
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "power";
    p.kinetic_eta_power = run.eta_grid_power;
    p.Nr = run.radial_cells;
    p.Nw = run.radial_velocity_cells;
    p.NL = run.eta_cells;
    p.r_min = run.radial_minimum;
    p.r_max = run.radial_maximum;
    p.radial_grid = "log";
    p.w_max = run.radial_velocity_maximum;
    p.L_max = run.angular_momentum_maximum;
    p.inner_boundary = "reflecting";
    p.outer_boundary = "reflecting";
    p.integrator = "semi_lagrangian_kick_drift_kick";
    p.bp_cross_section = run.sidm_cross_section;
    p.bp_boltzmann_kernel = run.sidm_cross_section == "yukawa_born"
        ? "yukawa_born"
        : "isotropic";
    p.bp_collision_strength = run.collision_strength;
    p.bp_sigma0 = run.sidm_sigma0;
    p.bp_sigma_u0 = run.sidm_u0;
    p.bp_sigma_n = run.sidm_power;
    p.bp_yukawa_alpha = run.yukawa_alpha;
    p.bp_yukawa_mchi = run.yukawa_mchi;
    p.bp_yukawa_mphi = run.yukawa_mphi;
    p.bp_reversible_candidate_radius = std::min(
        4, run.remap_candidate_radius);
    p.bp_reversible_projection_tolerance = 2.0e-10;
    return p;
}

struct CoreDiagnostic {
    double density = 0.0;
    double radial_variance = 0.0;
    double radius = 0.0;
};

// End time and output cadence can change on continuation. The physical model,
// mesh and evolution controls determine the continued trajectory and are
// recorded with the frozen field. Thread count fixes reduction/RNG ordering.
std::string frozen_field_contract(const RunOptions& run, const ev::Params& p) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "VENUS_STATIC_SIDM_FROZEN_FIELD_1\n" << std::hexfloat;
    auto add = [&](const char* name, const auto& value) {
        out << name << '=' << value << '\n';
    };
    add("maximum_timestep", run.maximum_timestep);
    add("dynamical_eta", run.dynamical_eta);
    add("potential_event_eta", run.potential_event_eta);
    add("collision_event_eta", run.collision_event_eta);
    add("radial_cells", run.radial_cells);
    add("radial_velocity_cells", run.radial_velocity_cells);
    add("eta_cells", run.eta_cells);
    add("radial_minimum", run.radial_minimum);
    add("radial_maximum", run.radial_maximum);
    add("radial_velocity_maximum", run.radial_velocity_maximum);
    add("angular_momentum_maximum", run.angular_momentum_maximum);
    add("eta_grid_power", run.eta_grid_power);
    add("remap_candidate_radius", run.remap_candidate_radius);
    add("maximum_carrier_mass_fraction", run.maximum_carrier_mass_fraction);
    add("phase_space_projection_at_events", run.phase_space_projection_at_events);
    add("threads", run.threads);
    add("shape", static_cast<int>(run.shape));
    add("cross_section", run.sidm_cross_section);
    add("collision_strength", run.collision_strength);
    add("sigma0", run.sidm_sigma0);
    add("u0", run.sidm_u0);
    add("power", run.sidm_power);
    add("yukawa_alpha", run.yukawa_alpha);
    add("yukawa_mchi", run.yukawa_mchi);
    add("yukawa_mphi", run.yukawa_mphi);
    add("collision_target_shell_markers", run.collision_target_shell_markers);
    add("collision_minimum_shell_markers", run.collision_minimum_shell_markers);
    add("collision_cfl", run.collision_cfl);
    add("collision_maximum_substeps", run.collision_maximum_substeps);
    add("collision_adaptive_shell_timestep", run.collision_adaptive_shell_timestep);
    add("collision_adaptive_maximum_interval", run.collision_adaptive_maximum_interval);
    add("fail_on_nonlocal_collision_shell", run.fail_on_nonlocal_collision_shell);
    add("gravity", p.vp_gravity);
    add("orbit_regularization", p.vp_orbit_regularization);
    add("orbit_energy_tolerance", p.vp_orbit_energy_tolerance);
    add("orbit_lc_relative_tolerance", p.vp_orbit_lc_relative_tolerance);
    add("orbit_lc_absolute_tolerance", p.vp_orbit_lc_absolute_tolerance);
    add("orbit_lc_max_steps", p.vp_orbit_lc_max_steps);
    add("orbit_lc_max_refinements", p.vp_orbit_lc_max_refinements);
    return out.str();
}

// A deliberate field refresh may change numerical mesh or orbit controls.
// Its pending collision age still belongs to the saved physical cross section
// and gravitational law, so require those physical parameters to agree.
bool same_restart_physics(const std::string& stored, const std::string& requested) {
    const std::string header = "VENUS_STATIC_SIDM_FROZEN_FIELD_1\n";
    if (stored.compare(0, header.size(), header) != 0) return false;
    auto value = [](const std::string& text, const char* name) {
        const std::string key = std::string("\n") + name + '=';
        const auto at = text.find(key);
        if (at == std::string::npos) return std::string();
        const auto begin = at + key.size();
        return text.substr(begin, text.find('\n', begin) - begin);
    };
    for (const char* key : {"cross_section", "collision_strength", "sigma0",
            "u0", "power", "yukawa_alpha", "yukawa_mchi", "yukawa_mphi", "gravity"}) {
        const auto old_value = value(stored, key);
        if (old_value.empty() || old_value != value(requested, key)) return false;
    }
    return true;
}

// Measure the core from a prescribed count of the innermost active particles.
// This is a particle-count definition: unequal particle masses need not give
// a fixed enclosed mass fraction. An existing current-radius index order can
// be reused; otherwise select the required indices without moving particles.
// ---- Core, energy, and profile diagnostics ---------------------------------
CoreDiagnostic innermost_core(
    const std::vector<ev::WeightedPICMarker>& markers,
    std::size_t requested = 0,
    const std::vector<std::size_t>* radius_order = nullptr) {
    std::vector<std::size_t> owned_active;
    const std::vector<std::size_t>* active = radius_order;
    if (active == nullptr) {
        owned_active.reserve(markers.size());
        for (std::size_t i = 0; i < markers.size(); ++i) {
            if (markers[i].state == ev::WeightedPICMarkerState::active) {
                owned_active.push_back(i);
            }
        }
        active = &owned_active;
    }
    if (requested == 0) {
        // Use at least 250 markers to control discreteness noise.  When the
        // imported state is subsampled, its enclosed mass fraction, rho_core,
        // and slope are not quantitatively comparable with a full import.
        requested = std::max<std::size_t>(
            250,
            static_cast<std::size_t>(std::llround(
                2.5e-4 * static_cast<double>(markers.size()))));
    }
    const std::size_t count = std::min(requested, active->size());
    if (count == 0) {
        return {};
    }
    if (radius_order == nullptr) {
        // Partition indices by the particles' CURRENT radii. The first count
        // indices identify the innermost particles but need no internal order.
        // Their identities, radii and all physical records remain unchanged.
        std::nth_element(
            owned_active.begin(),
            owned_active.begin() + static_cast<std::ptrdiff_t>(count - 1),
            owned_active.end(),
            [&](std::size_t first, std::size_t second) {
                return markers[first].r < markers[second].r;
            });
    }
    double radius = 0.0;
    long double mass = 0.0L;
    long double momentum = 0.0L;
    long double radial_second = 0.0L;
    for (std::size_t q = 0; q < count; ++q) {
        const ev::WeightedPICMarker& marker = markers[(*active)[q]];
        radius = std::max(radius, marker.r);
        mass += marker.mass;
        momentum += static_cast<long double>(marker.mass) * marker.w;
        radial_second += static_cast<long double>(marker.mass)
            * marker.w * marker.w;
    }
    // Subtract the squared mean radial velocity from <w^2> to measure random
    // radial motion rather than coherent infall. Divide the selected mass by
    // the sphere inside its outermost particle to define the core density.
    const double mean = static_cast<double>(momentum / mass);
    return {
        static_cast<double>(mass)
            / ((4.0 * ev::PI / 3.0) * radius * radius * radius),
        std::max(0.0, static_cast<double>(radial_second / mass) - mean * mean),
        radius,
    };
}

// Combine the extensive radial and tangential second moments into kinetic
// energy. The self-gravitational energy is half sum(m*Phi), because summing
// the potential over all matter counts every interacting pair twice.
double total_energy(
    const ev::WeightedPICDepositedMoments& deposit,
    const ev::VlasovPoissonField& field) {
    long double result = 0.0L;
    for (std::size_t i = 0; i < deposit.shell_mass.size(); ++i) {
        result += 0.5L
            * (deposit.shell_radial_second_moment[i]
               + deposit.shell_tangential_second_moment[i]);
        result += 0.5L * deposit.shell_mass[i] * field.phi[i];
    }
    return static_cast<double>(result);
}

// Convert mass-weighted deposited moments to local means and variances.
// The CSV stores physical radius and density with the effective particle
// count so a reader can assess how well each shell samples the distribution.
void write_profile(
    const std::filesystem::path& directory,
    std::int64_t step,
    double time,
    const ev::WeightedPICDepositedMoments& deposit,
    const ev::WeightedPICRadialMesh& mesh) {
    std::ostringstream name;
    name << "profile_" << std::setw(8) << std::setfill('0') << step << ".csv";
    std::ofstream output(directory / name.str());
    output << std::setprecision(17)
           << "time,radius,density,radial_mean,radial_variance,"
              "tangential_second,neff\n";
    for (std::size_t i = 0; i < deposit.density.size(); ++i) {
        const double density = deposit.density[i];
        const double mean = density > 0.0
            ? deposit.radial_momentum_density[i] / density
            : 0.0;
        // The deposited second moment contains both mean flow and dispersion.
        // Retain it here, then subtract mean^2 only in the variance column.
        const double raw = density > 0.0
            ? deposit.radial_second_moment_density[i] / density
            : 0.0;
        const double tangential = density > 0.0
            ? deposit.tangential_second_moment_density[i] / density
            : 0.0;
        output << time << ',' << mesh.venus_center[i] << ',' << density << ','
               << mean << ',' << std::max(0.0, raw - mean * mean) << ','
               << tangential << ',' << deposit.effective_marker_number[i] << '\n';
    }
}

}  // namespace

// ---- Frozen-field transport, NTC events, and restart loop ------------------
int ev::apps::run_static_sidm_weighted_vlasov_pic(
    int argc,
    char** argv) {
    // Stage 1: freeze the requested physics and construct the radial field,
    // phase-space geometry, and characteristic-orbit integration plan.
    const RunOptions run = parse_options(argc, argv);
    std::signal(SIGUSR1, request_orderly_stop);
    std::signal(SIGTERM, request_orderly_stop);
    const ev::Params p = make_parameters(run);
    std::cout << std::setprecision(17)
              << "SIDM kernel=" << p.bp_cross_section
              << " differential=" << p.bp_boltzmann_kernel
              << " strength=" << p.bp_collision_strength
              << " sigma0=" << p.bp_sigma0
              << " u0=" << p.bp_sigma_u0
              << " power=" << p.bp_sigma_n << '\n';
    // Gravity is reconstructed on radial cells. The separate (r,w,eta)
    // geometry is needed when projecting particles onto a phase-space grid.
    const ev::WeightedPICFieldGrid grid =
        ev::make_weighted_pic_field_grid(p, p.Nr);
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    const ev::CharacteristicIntegratorPlan orbit_plan =
        ev::make_characteristic_integrator_plan(p);
    ev::WeightedPICRestartState checkpoint;
    // Stage 2: restore a complete weighted-PIC state, or convert a Frank
    // spherical realization, including unequal weights, into markers
    // representing the same total mass.
    if (!run.pic_restart.empty()) {
        checkpoint = ev::read_weighted_pic_restart(run.pic_restart);
    } else {
        ev::WeightedPICFrankRestartImport imported =
            ev::import_weighted_pic_frank_restart(
                run.frank_restart, run.marker_limit);
        checkpoint.markers = std::move(imported.markers);
        checkpoint.time = imported.time;
        checkpoint.step = imported.step;
        checkpoint.sampler = {UINT64_C(0x6672616e6b706963), 0};
        checkpoint.collision_sampler = {UINT64_C(5489), 0};
    }
    if (checkpoint.time >= run.end_time) {
        throw std::runtime_error("The requested end time is not after restart time.");
    }
    // Compare the requested controls with the settings saved beside the
    // frozen field. An ordinary continuation must keep that field and its
    // event ages; rebuilding it is a separate explicit numerical choice.
    const std::string restart_contract = frozen_field_contract(run, p);
    if (checkpoint.has_frozen_field && run.refresh_restart_field
        && !same_restart_physics(checkpoint.frozen_field_contract, restart_contract)) {
        throw std::runtime_error(
            "A restart field refresh requires the saved physical gravity and "
            "collision laws. Pending collision time belongs to that physical model.");
    }
    if (checkpoint.has_frozen_field && !run.refresh_restart_field
        && checkpoint.frozen_field_contract != restart_contract) {
        throw std::runtime_error(
            "Static SIDM restart evolution controls differ; repeat the saved "
            "physical, mesh, orbit, collision and thread settings, or explicitly "
            "select --refresh-restart-field for a new field interval.");
    }
    // Appending observations later than the restored state would combine two
    // different trajectories in one history. Check chronology before writing.
    ev::require_output_precedes_restart(run.output_directory, checkpoint.time,
        {"history.csv", "operator_mass_budget.csv"});
    std::filesystem::create_directories(run.output_directory);
    const bool new_history = run.pic_restart.empty()
        || !std::filesystem::exists(run.output_directory / "history.csv");
    std::ofstream history(
        run.output_directory / "history.csv",
        new_history ? std::ios::trunc : std::ios::app);
    if (!history) {
        throw std::runtime_error("Could not create weighted PIC Frank history.");
    }
    if (new_history) {
        history << "step,time,dt,core_density,core_sigma_r2,core_radius,"
                   "mass,total_energy,collision_trials,collision_accepted,"
                   "collision_substeps,collision_shell_visits,"
                   "max_shell_width_over_mfp,nonlocal_shell_visits,"
                   "max_occupied_shell_span_over_mfp,"
                   "nonlocal_occupied_shell_visits,"
                   "mfp_pair_samples,min_orbital_sigma,max_orbital_sigma,"
                   "collision_radial_momentum_error,collision_kinetic_error\n";
    }

    const bool new_budget = run.pic_restart.empty()
        || !std::filesystem::exists(
            run.output_directory / "operator_mass_budget.csv");
    std::ofstream reconstruction_budget(
        run.output_directory / "operator_mass_budget.csv",
        new_budget ? std::ios::trunc : std::ios::app);
    if (!reconstruction_budget) {
        throw std::runtime_error(
            "Could not create weighted PIC reconstruction budget.");
    }
    if (new_budget) {
        reconstruction_budget
            << "step,time,event_order,reconstruction_count,"
               "reconstructed_before_collision,reconstructed_after_collision,"
               "marker_mass_before,marker_mass_after,active_remap_mass,"
               "grid_remap_mass,carrier_mass,carrier_fraction,"
               "radial_momentum_error,kinetic_energy_error,"
               "maximum_projection_error\n";
    }

    ev::WeightedPICCollisionOptions collision_options;
    // Stage 3: configure the two physical event clocks.  Potential events
    // refresh self-gravity after the mass profile evolves; collision events
    // apply the local no-time-counter (NTC) Monte Carlo collision generator.
    collision_options.target_shell_markers =
        run.collision_target_shell_markers;
    collision_options.minimum_shell_markers =
        run.collision_minimum_shell_markers;
    collision_options.collision_cfl = run.collision_cfl;
    collision_options.maximum_substeps = run.collision_maximum_substeps;
    collision_options.adaptive_shell_timestep =
        run.collision_adaptive_shell_timestep;
    collision_options.fail_on_nonlocal_shell =
        run.fail_on_nonlocal_collision_shell;
    ev::WeightedPICBoltzmannEventOptions event_options;
    event_options.potential_fraction = run.potential_event_eta;
    event_options.collision_fraction = run.collision_event_eta;
    event_options.maximum_orbit_interval = run.maximum_timestep;
    event_options.adaptive_collision_shell_timestep =
        run.collision_adaptive_shell_timestep;
    event_options.adaptive_collision_maximum_interval =
        run.collision_adaptive_maximum_interval;
    event_options.threads = run.threads;
    event_options.field_shape = run.shape;
    event_options.phase_space_projection_at_events =
        run.phase_space_projection_at_events;
    event_options.phase_remap.maximum_candidate_radius =
        run.remap_candidate_radius;
    event_options.phase_remap.maximum_carrier_mass_fraction =
        run.maximum_carrier_mass_fraction;

    // Keep initialization and collision random streams distinct. Restoring
    // the particle list alone would lose where the encounter sequence paused.
    const ev::WeightedPICSamplerState marker_sampler = checkpoint.sampler;
    ev::WeightedPICBoltzmannEventState state =
        ev::initialize_weighted_pic_boltzmann_event_state(
            std::move(checkpoint.markers),
            checkpoint.time,
            checkpoint.step,
            {checkpoint.potential_event_age, checkpoint.collision_event_age},
            checkpoint.collision_sampler,
            grid,
            geometry,
            p,
            event_options,
            collision_options);
    if (!run.pic_restart.empty()) {
        bool restored = false;
        if (run.refresh_restart_field) {
            // A rebuilt potential starts a new field interval. The elapsed
            // collision age and collision random state remain the saved ones.
            state.clock.potential_age = 0.0;
            std::cerr << "Explicit restart field refresh: starting a new potential "
                         "interval with the requested evolution controls.\n";
        } else {
            restored = ev::restore_weighted_pic_boltzmann_frozen_field(
                state, checkpoint, restart_contract);
        }
        if (!restored && !run.refresh_restart_field) {
            std::cerr << "Legacy particle checkpoint: rebuilding the frozen "
                         "Poisson field and event timescales at import.\n";
        }
        std::ofstream import(run.output_directory / "restart_import.txt");
        import << std::setprecision(17) << "source=" << run.pic_restart.string()
               << "\ntime=" << state.time << "\nstep=" << state.step
               << "\nfrozen_field_restored=" << restored
               << "\nexplicit_field_refresh=" << run.refresh_restart_field << '\n'
               << restart_contract;
        if (!import) throw std::runtime_error("Could not record SIDM restart import.");
    }
    const double represented_mass =
        ev::weighted_pic_total_marker_mass(state.markers);
    // Record the imported initial state before the first event of this run.
    // Its physical time is the time carried by the source particle file.
    if (run.pic_restart.empty()) {
        const CoreDiagnostic initial_core = innermost_core(
            state.markers,
            0,
            &state.collision_workspace.active_by_radius);
        history << std::setprecision(17)
                << state.step << ',' << state.time << ",0,"
                << initial_core.density << ','
                << initial_core.radial_variance << ','
                << initial_core.radius << ','
                << state.deposited.deposited_mesh_mass << ','
                << total_energy(state.deposited, state.field)
                << ",0,0,0,0,0,0,0,0,0,0,0,0,0\n";
        history.flush();
        write_profile(
            run.output_directory,
            state.step,
            state.time,
            state.deposited,
            grid.mesh);
    }

    auto write_reconstruction_budget = [&]
        (const ev::WeightedPICBoltzmannEventDiagnostics& diagnostic,
         int event_order) {
        if (diagnostic.reconstruction_count == 0) {
            return;
        }
        const ev::WeightedPICPhaseRemapDiagnostics& remap =
            diagnostic.reconstruction;
        // Split the represented mass between the gridded part and particles
        // retained continuously. Their sum must recover the active mass.
        const double grid_mass = remap.active_mass - remap.carrier_mass;
        const double carrier_fraction = remap.active_mass > 0.0
            ? remap.carrier_mass / remap.active_mass
            : 0.0;
        reconstruction_budget << std::setprecision(17)
            << state.step << ',' << state.time << ',' << event_order << ','
            << diagnostic.reconstruction_count << ','
            << (diagnostic.reconstructed_before_collision ? 1 : 0) << ','
            << (diagnostic.reconstructed_after_collision ? 1 : 0) << ','
            << diagnostic.marker_mass_before << ','
            << diagnostic.marker_mass_after << ','
            << remap.active_mass << ',' << grid_mass << ','
            << remap.carrier_mass << ',' << carrier_fraction << ','
            << remap.radial_momentum_after - remap.radial_momentum_before << ','
            << remap.kinetic_energy_after - remap.kinetic_energy_before << ','
            << remap.maximum_projection_error << '\n';
        reconstruction_budget.flush();
    };
    auto write_checkpoint = [&] {
        // Store marker phase coordinates, weights, RNG counters, and event
        // ages together; omitting any of them would change a resumed history.
        ev::WeightedPICRestartState output;
        output.sampler = marker_sampler;
        output.collision_sampler = state.collision_sampler;
        output.potential_event_age = state.clock.potential_age;
        output.collision_event_age = state.clock.collision_age;
        output.potential_timescale_estimate = state.potential_timescale;
        output.collision_timescale_estimate = state.collision_timescale;
        output.has_frozen_field = true;
        output.frozen_field = state.field;
        output.frozen_field_contract = restart_contract;
        output.time = state.time;
        output.step = state.step;
        // Marker identities are validated when the initial state is loaded
        // and collisions preserve them.  Temporarily exchanging the vectors
        // avoids copying the full particle realization and repeating an
        // O(N log N) identity sort at every checkpoint.
        output.markers.swap(state.markers);
        try {
            ev::write_weighted_pic_restart(
                run.output_directory / "restart_state.bin", output, false);
        } catch (...) {
            output.markers.swap(state.markers);
            throw;
        }
        output.markers.swap(state.markers);
    };

    const auto start = std::chrono::steady_clock::now();
    const std::int64_t initial_step = state.step;
    // Stage 4: alternate frozen-field characteristic motion with whichever
    // potential/collision deadline occurs first.  The event routine performs
    // conservative reconstruction whenever the two representations must be
    // synchronized.
    while (state.time < run.end_time && stop_requested == 0) {
        const ev::WeightedPICBoltzmannEventDiagnostics diagnostic =
            ev::advance_weighted_pic_boltzmann_event(
                state,
                grid,
                geometry,
                p,
                orbit_plan,
                event_options,
                collision_options,
                run.end_time - state.time);
        // The event planner returns the actual time spent on orbit motion.
        // A field update already due at the current time may have dt=0 while
        // still performing a meaningful reconstruction or collision update.
        const double dt = diagnostic.plan.orbit_interval;
        write_reconstruction_budget(
            diagnostic, static_cast<int>(diagnostic.plan.order));

        if (state.step % run.output_every == 0 || state.time == run.end_time) {
            // Every event leaves these radial moments current.  Reusing them
            // avoids a second all-marker deposit at each profile output.
            const ev::WeightedPICDepositedMoments& deposit = state.deposited;
            // Reconstruct gravity for the output energy diagnostic from the
            // current moments. The event state separately owns the frozen
            // field used to integrate the particle trajectories.
            const ev::VlasovPoissonField field =
                ev::solve_weighted_pic_poisson_field(
                    deposit, grid.mesh, state.time, p);
            const CoreDiagnostic core = innermost_core(
                state.markers,
                0,
                &state.collision_workspace.active_by_radius);
            history << std::setprecision(17)
                    << state.step << ',' << state.time << ',' << dt << ','
                    << core.density << ',' << core.radial_variance << ','
                    << core.radius << ',' << deposit.deposited_mesh_mass << ','
                    << total_energy(deposit, field) << ','
                    << diagnostic.collision.trial_pairs << ','
                    << diagnostic.collision.accepted_pairs << ','
                    << diagnostic.collision.substeps << ','
                    << diagnostic.collision.shell_visits << ','
                    << diagnostic.collision.maximum_shell_width_over_mean_free_path << ','
                    << diagnostic.collision.nonlocal_shells << ','
                    << diagnostic.collision.maximum_occupied_shell_span_over_mean_free_path << ','
                    << diagnostic.collision.nonlocal_occupied_shells << ','
                    << diagnostic.collision.mean_free_path_pair_samples << ','
                    << diagnostic.collision.minimum_orbital_velocity_dispersion << ','
                    << diagnostic.collision.maximum_orbital_velocity_dispersion << ','
                    << diagnostic.collision.radial_momentum_error << ','
                    << diagnostic.collision.kinetic_energy_error << '\n';
            history.flush();
            write_profile(
                run.output_directory, state.step, state.time, deposit, grid.mesh);
            std::cout << std::setprecision(10)
                      << "step=" << state.step << " t=" << state.time
                      << " dt=" << dt << " rho_core=" << core.density
                      << " sigma_r2_core=" << core.radial_variance
                      << " accepted=" << diagnostic.collision.accepted_pairs
                      << " event=" << static_cast<int>(diagnostic.plan.order)
                      << " age_phi=" << state.clock.potential_age
                      << " age_coll=" << state.clock.collision_age
                      << " locality="
                      << diagnostic.collision.maximum_shell_width_over_mean_free_path
                      << " nonlocal=" << diagnostic.collision.nonlocal_shells
                      << " locality_warning="
                      << (diagnostic.collision.maximum_shell_width_over_mean_free_path
                              >= 1.0
                              ? 1
                              : 0)
                      << " orbit_s=" << diagnostic.orbit_seconds
                      << " deposit_field_s="
                      << diagnostic.deposition_and_field_seconds
                      << " collision_s=" << diagnostic.collision_seconds
                      << " clocks_s=" << diagnostic.clock_refresh_seconds
                      << " carrier_mass="
                      << state.reconstruction_carrier_mass
                      << " reconstruction_current="
                      << (state.reconstructed_f_current ? 1 : 0) << '\n'
                      << std::flush;
        }
        if (state.step % run.restart_every == 0 || state.time == run.end_time) {
            write_checkpoint();
        }
    }
    const double event_loop_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    const std::int64_t completed_events = state.step - initial_step;
    std::cout << std::setprecision(10)
              << "physical_event_loop_seconds=" << event_loop_seconds
              << " completed_events=" << completed_events
              << " seconds_per_event="
              << (completed_events > 0
                      ? event_loop_seconds / static_cast<double>(completed_events)
                      : 0.0)
              << '\n'
              << std::flush;
    const bool stopped_for_walltime = stop_requested != 0;
    if (stopped_for_walltime) {
        // Slurm sends SIGUSR1 before the allocation ends.  Finish the current
        // indivisible physical event, then store the complete marker and
        // random-clock state.  A resumed trajectory therefore starts from the
        // same kinetic state as an uninterrupted run.
        write_checkpoint();
        history.flush();
        reconstruction_budget.flush();
        std::cout << std::setprecision(17)
                  << "received walltime signal after step=" << state.step
                  << " t=" << state.time
                  << "; restart checkpoint completed\n";
    }
    if (!stopped_for_walltime
        && run.phase_space_projection_at_final_output
        && !state.reconstructed_f_current) {
        // Particle coordinates can be current while the optional phase-space
        // grid still represents an earlier event. Project the particles once
        // before finishing and record the mass and moment reconstruction.
        ev::WeightedPICBoltzmannEventDiagnostics final_sync;
        final_sync.marker_mass_before =
            ev::weighted_pic_total_marker_mass(state.markers);
        final_sync.reconstruction =
            ev::synchronize_weighted_pic_boltzmann_distribution(
                state, geometry, p, event_options.phase_remap);
        final_sync.reconstruction_count = 1;
        final_sync.marker_mass_after = final_sync.marker_mass_before;
        write_reconstruction_budget(final_sync, -1);
    }
    const double final_mass = ev::weighted_pic_total_marker_mass(state.markers);
    // Stage 5: enforce the strongest global invariant of elastic scattering.
    // Momentum and kinetic-energy residuals are recorded event by event;
    // represented mass must agree to roundoff at the end of the run.
    if (std::abs(final_mass - represented_mass)
        > 128.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, represented_mass)) {
        throw std::runtime_error("Static SIDM WPIC run changed marker mass.");
    }
    std::cout << "Static SIDM WPIC run "
              << (stopped_for_walltime ? "checkpointed" : "completed")
              << " in "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start).count()
              << " seconds\n";
    return 0;
}
