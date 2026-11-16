// Evolve a spherical Boltzmann--Poisson distribution on a mesh that follows
// bound orbits.  Each row has fixed eta=L^2/2 and a range of orbital energies.
// Its phase cells record where matter lies along the radial orbit.  Transport
// moves this matter along the orbit, while encounters change its velocity and
// therefore its energy, angular momentum and destination row.
//
// Inputs are a spherical DSMC restart, an Eddington distribution f(H), and
// numerical controls for the orbit mesh and collision sampling.  The driver
// first reconstructs the density and gravitational potential, fills the orbit
// mesh, and checks how much of the original distribution it represents.  It
// then advances the field and collision events with their separate clocks.
// Outputs contain radial profiles, energy and operator mass budgets, and a
// restart containing the orbit populations, collision carriers and clocks.
//
// The reusable projection, transport, encounter and restart calculations are
// in include/ev.  This file connects them and explains the order in which the
// physical state is assembled and advanced.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include "ev/config.hpp"
#include "ev/canonical_boltzmann_poisson_event.hpp"
#include "ev/canonical_boltzmann_poisson_restart.hpp"
#include "ev/canonical_orbit_phase_transport.hpp"
#include "ev/solver_setup.hpp"

namespace {

// The first controls describe the orbit mesh and its quadrature.  Collision
// controls set the sampling population and encounter interval.  Output and
// restart cadences count accepted events, each of which may contain a field
// update, a collision update, or both.
struct RunOptions {
    std::filesystem::path restart;
    std::filesystem::path eddington_table;
    std::filesystem::path output_directory;
    std::filesystem::path canonical_restart;
    int energy_rows = 8;
    int phase_cells = 16;
    int boundary_segments = 12;
    int mapping_energy_points = 8;
    int mapping_phase_points = 8;
    int atlas_eta_points = 2;
    double energy_power = 1.0;
    double transport_time = 0.1;
    double collision_time = 0.0;
    double collision_strength = 1.0;
    std::size_t collision_markers = 65536;
    int collision_markers_per_occupied_cell = 32;
    std::size_t collision_target_shell_markers = 4096;
    std::size_t collision_minimum_shell_markers = 16;
    std::size_t open_compression_trigger = 65536;
    int open_compression_subcells = 1;
    double open_compression_tolerance = 2.0e-11;
    double source_row_fraction = 0.5;
    double minimum_energy_guard_fraction = 1.0e-6;
    double maximum_lift_phase_cell_error = 0.1;
    int collision_batches = 1;
    double minimum_shell_trials = 0.0;
    int maximum_collision_batches = 4096;
    double core_importance_radius = 0.0;
    double core_importance_fraction = 0.0;
    int collision_convergence_levels = 0;
    int collision_marker_convergence_levels = 0;
    int collision_balanced_convergence_levels = 0;
    int reconstruction_cycles = 0;
    int evolution_events = 0;
    double evolution_time = 0.0;
    int output_every = 1;
    int restart_every = 10;
    double event_dt = 0.1;
    int event_dt_convergence_levels = 0;
    double event_dt_convergence_time = 0.0;
    double potential_fraction = 0.05;
    double collision_fraction = 0.1;
    double collision_cfl_safety = 0.5;
    double maximum_open_mass_fraction = 1.0e-3;
    double maximum_collision_excluded_open_mass_fraction = 1.0e-5;
    double maximum_detached_mass_fraction = 1.0;
    bool absorb_detached_carriers_on_field_rebuild = false;
    double maximum_unrepresented_mass_fraction = 1.0e-8;
    double maximum_collision_escaped_increment_fraction = 0.0;
    double maximum_collision_moment_repair_mass_fraction = 2.0e-12;
    double startup_field_tolerance = 2.0e-2;
    int startup_field_iterations = 0;
    std::string atlas_initialization = "analytic";
    std::string collision_coupling = "carrier_ntc";
    std::string reversible_operator = "streamed";
    int sampled_dense_cells = 48;
    int sampled_min_samples = 2048;
    int sampled_max_samples = 262144;
    double sampled_guided_fraction = 0.5;
    double sampled_rhs_tolerance = 0.01;
    double sampled_heat_tolerance = 0.01;
    bool sampled_fail_on_unconverged = true;
    bool direct_fh_diagnostic_only = false;
    double collision_locality_max_ratio = 1.0;
    double collision_cfl = 0.15;
    unsigned int threads = 4;
    int nr = 48;
    int nw = 32;
    int nl = 16;
    double r_min = 1.0e-3;
    double r_max = 1000.0;
    double w_max = 3.0;
    double L_max = 100.0;
    std::string eta_grid = "power";
    double eta_power = 10.0;
    double eta_join = 0.0;
    int eta_inner_cells = 0;
};

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " --restart PATH [options]\n"
        << "\nRequired initial state:\n"
        << "  --restart PATH                 source SphDSMC restart\n"
        << "  --eddington-table PATH         optional stationary f(H) table\n"
        << "  --canonical-restart PATH       resume a canonical-atlas state\n"
        << "\nAtlas and physical evolution:\n"
        << "  --energy-rows N --phase-cells N --atlas-eta-points N\n"
        << "  --evolution-time T | --evolution-events N\n"
        << "  --event-dt DT --potential-fraction X --collision-fraction X\n"
        << "  --collision-strength S --collision-coupling NAME\n"
        << "  --reversible-operator {exact|reversible|sampled|streamed}\n"
        << "\nGrid, output, and execution:\n"
        << "  --nr N --nw N --nl N --r-min R --r-max R --w-max W --L-max L\n"
        << "  --output-directory DIR --output-every N --restart-every N\n"
        << "  --threads N\n"
        << "\nAdvanced atlas, compression, sampling, locality, and convergence\n"
        << "controls are described in the VENUS manual.\n"
        << "  --help, -h                     show this summary\n";
}

// ---- Atlas, event, collision, compression, and restart options ------------
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
        } else if (name == "--restart") {
            result.restart = value(argument, "--restart");
        } else if (name == "--eddington-table") {
            result.eddington_table = value(argument, "--eddington-table");
        } else if (name == "--output-directory") {
            result.output_directory = value(argument, "--output-directory");
        } else if (name == "--canonical-restart") {
            result.canonical_restart = value(
                argument, "--canonical-restart");
        } else if (name == "--energy-rows") {
            result.energy_rows = std::stoi(value(argument, "--energy-rows"));
        } else if (name == "--phase-cells") {
            result.phase_cells = std::stoi(value(argument, "--phase-cells"));
        } else if (name == "--boundary-segments") {
            result.boundary_segments = std::stoi(
                value(argument, "--boundary-segments"));
        } else if (name == "--mapping-energy-points") {
            result.mapping_energy_points = std::stoi(
                value(argument, "--mapping-energy-points"));
        } else if (name == "--mapping-phase-points") {
            result.mapping_phase_points = std::stoi(
                value(argument, "--mapping-phase-points"));
        } else if (name == "--atlas-eta-points") {
            result.atlas_eta_points = std::stoi(
                value(argument, "--atlas-eta-points"));
        } else if (name == "--energy-power") {
            result.energy_power = std::stod(value(argument, "--energy-power"));
        } else if (name == "--transport-time") {
            result.transport_time = std::stod(
                value(argument, "--transport-time"));
        } else if (name == "--collision-time") {
            result.collision_time = std::stod(
                value(argument, "--collision-time"));
        } else if (name == "--collision-strength") {
            result.collision_strength = std::stod(
                value(argument, "--collision-strength"));
        } else if (name == "--collision-markers") {
            result.collision_markers = static_cast<std::size_t>(
                std::stoull(value(argument, "--collision-markers")));
        } else if (name == "--collision-markers-per-occupied-cell") {
            result.collision_markers_per_occupied_cell = std::stoi(
                value(argument, "--collision-markers-per-occupied-cell"));
        } else if (name == "--collision-target-shell-markers") {
            result.collision_target_shell_markers =
                static_cast<std::size_t>(std::stoull(value(
                    argument, "--collision-target-shell-markers")));
        } else if (name == "--collision-minimum-shell-markers") {
            result.collision_minimum_shell_markers =
                static_cast<std::size_t>(std::stoull(value(
                    argument, "--collision-minimum-shell-markers")));
        } else if (name == "--open-compression-trigger") {
            result.open_compression_trigger = static_cast<std::size_t>(
                std::stoull(value(argument, "--open-compression-trigger")));
        } else if (name == "--open-compression-subcells") {
            result.open_compression_subcells = std::stoi(
                value(argument, "--open-compression-subcells"));
        } else if (name == "--open-compression-tolerance") {
            result.open_compression_tolerance = std::stod(
                value(argument, "--open-compression-tolerance"));
        } else if (name == "--source-row-fraction") {
            result.source_row_fraction = std::stod(
                value(argument, "--source-row-fraction"));
        } else if (name == "--minimum-energy-guard-fraction") {
            result.minimum_energy_guard_fraction = std::stod(
                value(argument, "--minimum-energy-guard-fraction"));
        } else if (name == "--maximum-lift-phase-cell-error") {
            result.maximum_lift_phase_cell_error = std::stod(
                value(argument, "--maximum-lift-phase-cell-error"));
        } else if (name == "--collision-batches") {
            result.collision_batches = std::stoi(
                value(argument, "--collision-batches"));
        } else if (name == "--minimum-shell-trials") {
            result.minimum_shell_trials = std::stod(
                value(argument, "--minimum-shell-trials"));
        } else if (name == "--maximum-collision-batches") {
            result.maximum_collision_batches = std::stoi(
                value(argument, "--maximum-collision-batches"));
        } else if (name == "--core-importance-radius") {
            result.core_importance_radius = std::stod(
                value(argument, "--core-importance-radius"));
        } else if (name == "--core-importance-fraction") {
            result.core_importance_fraction = std::stod(
                value(argument, "--core-importance-fraction"));
        } else if (name == "--collision-convergence-levels") {
            result.collision_convergence_levels = std::stoi(
                value(argument, "--collision-convergence-levels"));
        } else if (name == "--collision-marker-convergence-levels") {
            result.collision_marker_convergence_levels = std::stoi(
                value(argument, "--collision-marker-convergence-levels"));
        } else if (name == "--collision-balanced-convergence-levels") {
            result.collision_balanced_convergence_levels = std::stoi(
                value(argument, "--collision-balanced-convergence-levels"));
        } else if (name == "--reconstruction-cycles") {
            result.reconstruction_cycles = std::stoi(
                value(argument, "--reconstruction-cycles"));
        } else if (name == "--evolution-events") {
            result.evolution_events = std::stoi(
                value(argument, "--evolution-events"));
        } else if (name == "--evolution-time") {
            result.evolution_time = std::stod(
                value(argument, "--evolution-time"));
        } else if (name == "--output-every") {
            result.output_every = std::stoi(
                value(argument, "--output-every"));
        } else if (name == "--restart-every") {
            result.restart_every = std::stoi(
                value(argument, "--restart-every"));
        } else if (name == "--event-dt") {
            result.event_dt = std::stod(value(argument, "--event-dt"));
        } else if (name == "--event-dt-convergence-levels") {
            result.event_dt_convergence_levels = std::stoi(
                value(argument, "--event-dt-convergence-levels"));
        } else if (name == "--event-dt-convergence-time") {
            result.event_dt_convergence_time = std::stod(
                value(argument, "--event-dt-convergence-time"));
        } else if (name == "--potential-fraction") {
            result.potential_fraction = std::stod(
                value(argument, "--potential-fraction"));
        } else if (name == "--collision-fraction") {
            result.collision_fraction = std::stod(
                value(argument, "--collision-fraction"));
        } else if (name == "--collision-cfl-safety") {
            result.collision_cfl_safety = std::stod(
                value(argument, "--collision-cfl-safety"));
        } else if (name == "--maximum-open-mass-fraction") {
            result.maximum_open_mass_fraction = std::stod(
                value(argument, "--maximum-open-mass-fraction"));
        } else if (name
                   == "--maximum-collision-excluded-open-mass-fraction") {
            result.maximum_collision_excluded_open_mass_fraction = std::stod(
                value(
                    argument,
                    "--maximum-collision-excluded-open-mass-fraction"));
        } else if (name == "--maximum-detached-mass-fraction") {
            result.maximum_detached_mass_fraction = std::stod(
                value(argument, "--maximum-detached-mass-fraction"));
        } else if (name
                   == "--absorb-detached-carriers-on-field-rebuild") {
            const std::string setting = value(
                argument,
                "--absorb-detached-carriers-on-field-rebuild");
            result.absorb_detached_carriers_on_field_rebuild =
                setting == "true" || setting == "1";
        } else if (name == "--maximum-unrepresented-mass-fraction") {
            result.maximum_unrepresented_mass_fraction = std::stod(
                value(argument, "--maximum-unrepresented-mass-fraction"));
        } else if (name
                   == "--maximum-collision-escaped-increment-fraction") {
            result.maximum_collision_escaped_increment_fraction = std::stod(
                value(
                    argument,
                    "--maximum-collision-escaped-increment-fraction"));
        } else if (name
                   == "--maximum-collision-moment-repair-mass-fraction") {
            result.maximum_collision_moment_repair_mass_fraction = std::stod(
                value(
                    argument,
                    "--maximum-collision-moment-repair-mass-fraction"));
        } else if (name == "--startup-field-tolerance") {
            result.startup_field_tolerance = std::stod(
                value(argument, "--startup-field-tolerance"));
        } else if (name == "--startup-field-iterations") {
            result.startup_field_iterations = std::stoi(
                value(argument, "--startup-field-iterations"));
        } else if (name == "--atlas-initialization") {
            result.atlas_initialization = value(
                argument, "--atlas-initialization");
        } else if (name == "--collision-coupling") {
            result.collision_coupling = value(
                argument, "--collision-coupling");
        } else if (name == "--reversible-operator") {
            result.reversible_operator = value(
                argument, "--reversible-operator");
        } else if (name == "--sampled-dense-cells") {
            result.sampled_dense_cells = std::stoi(
                value(argument, "--sampled-dense-cells"));
        } else if (name == "--sampled-min-samples") {
            result.sampled_min_samples = std::stoi(
                value(argument, "--sampled-min-samples"));
        } else if (name == "--sampled-max-samples") {
            result.sampled_max_samples = std::stoi(
                value(argument, "--sampled-max-samples"));
        } else if (name == "--sampled-guided-fraction") {
            result.sampled_guided_fraction = std::stod(
                value(argument, "--sampled-guided-fraction"));
        } else if (name == "--sampled-rhs-tolerance") {
            result.sampled_rhs_tolerance = std::stod(
                value(argument, "--sampled-rhs-tolerance"));
        } else if (name == "--sampled-heat-tolerance") {
            result.sampled_heat_tolerance = std::stod(
                value(argument, "--sampled-heat-tolerance"));
        } else if (name == "--sampled-fail-on-unconverged") {
            const std::string setting = value(
                argument, "--sampled-fail-on-unconverged");
            if (setting != "true" && setting != "false"
                && setting != "1" && setting != "0") {
                throw std::runtime_error(
                    "--sampled-fail-on-unconverged expects true or false.");
            }
            result.sampled_fail_on_unconverged =
                setting == "true" || setting == "1";
        } else if (name == "--direct-fh-diagnostic-only") {
            const std::string setting = value(
                argument, "--direct-fh-diagnostic-only");
            if (setting != "true" && setting != "false"
                && setting != "1" && setting != "0") {
                throw std::runtime_error(
                    "--direct-fh-diagnostic-only expects true or false.");
            }
            result.direct_fh_diagnostic_only =
                setting == "true" || setting == "1";
        } else if (name == "--collision-locality-max-ratio") {
            result.collision_locality_max_ratio = std::stod(
                value(argument, "--collision-locality-max-ratio"));
        } else if (name == "--collision-cfl") {
            result.collision_cfl = std::stod(
                value(argument, "--collision-cfl"));
        } else if (name == "--threads") {
            result.threads = static_cast<unsigned int>(
                std::stoul(value(argument, "--threads")));
        } else if (name == "--nr") {
            result.nr = std::stoi(value(argument, "--nr"));
        } else if (name == "--nw") {
            result.nw = std::stoi(value(argument, "--nw"));
        } else if (name == "--nl") {
            result.nl = std::stoi(value(argument, "--nl"));
        } else if (name == "--r-min") {
            result.r_min = std::stod(value(argument, "--r-min"));
        } else if (name == "--r-max") {
            result.r_max = std::stod(value(argument, "--r-max"));
        } else if (name == "--w-max") {
            result.w_max = std::stod(value(argument, "--w-max"));
        } else if (name == "--L-max") {
            result.L_max = std::stod(value(argument, "--L-max"));
        } else if (name == "--eta-grid") {
            result.eta_grid = value(argument, "--eta-grid");
        } else if (name == "--eta-power") {
            result.eta_power = std::stod(value(argument, "--eta-power"));
        } else if (name == "--eta-join") {
            result.eta_join = std::stod(value(argument, "--eta-join"));
        } else if (name == "--eta-inner-cells") {
            result.eta_inner_cells = std::stoi(
                value(argument, "--eta-inner-cells"));
        } else {
            throw std::runtime_error(
                "Unknown canonical-orbit option: " + name);
        }
    }
    // Validate the requested quadrature, populations and fractional changes
    // before constructing a state.  These bounds keep the corresponding mesh
    // and sampling operations within the ranges implemented by the modules.
    if (result.restart.empty() || result.energy_rows < 1
        || result.phase_cells < 4 || result.boundary_segments < 4
        || result.mapping_energy_points < 1
        || result.mapping_energy_points > 12
        || result.mapping_phase_points < 1
        || result.mapping_phase_points > 12
        || result.atlas_eta_points < 1
        || result.atlas_eta_points > 12
        || !(result.energy_power > 0.0) || !std::isfinite(result.energy_power)
        || !(result.transport_time >= 0.0)
        || !std::isfinite(result.transport_time)
        || !(result.collision_time >= 0.0)
        || !std::isfinite(result.collision_time)
        || !(result.collision_strength >= 0.0)
        || !std::isfinite(result.collision_strength)
        || result.collision_markers < 1024
        || result.collision_markers_per_occupied_cell < 1
        || result.collision_target_shell_markers < 2
        || result.collision_minimum_shell_markers < 2
        || result.collision_minimum_shell_markers
            > result.collision_target_shell_markers
        || !(result.source_row_fraction > 0.0)
        || !std::isfinite(result.source_row_fraction)
        || !(result.minimum_energy_guard_fraction > 0.0)
        || !(result.minimum_energy_guard_fraction < 1.0e-2)
        || !std::isfinite(result.minimum_energy_guard_fraction)
        || !(result.maximum_lift_phase_cell_error > 0.0)
        || !std::isfinite(result.maximum_lift_phase_cell_error)
        || result.collision_batches < 1
        || !(result.minimum_shell_trials >= 0.0)
        || !std::isfinite(result.minimum_shell_trials)
        || result.maximum_collision_batches < result.collision_batches
        || result.maximum_collision_batches > 4096
        || !(result.core_importance_radius >= 0.0)
        || !std::isfinite(result.core_importance_radius)
        || !(result.core_importance_fraction >= 0.0)
        || !(result.core_importance_fraction <= 1.0)
        || !std::isfinite(result.core_importance_fraction)
        || result.collision_convergence_levels < 0
        || result.collision_convergence_levels > 8
        || result.collision_marker_convergence_levels < 0
        || result.collision_marker_convergence_levels > 8
        || result.collision_balanced_convergence_levels < 0
        || result.collision_balanced_convergence_levels > 8
        || (result.collision_balanced_convergence_levels > 0
            && result.collision_batches
                < (1 << (result.collision_balanced_convergence_levels - 1)))
        || (static_cast<std::uint64_t>(result.collision_batches)
            << std::max(0, result.collision_convergence_levels - 1)) > 4096
        || result.reconstruction_cycles < 0
        || result.evolution_events < 0
        || !(result.evolution_time >= 0.0)
        || !std::isfinite(result.evolution_time)
        || result.output_every < 1
        || result.restart_every < 1
        || !(result.event_dt > 0.0) || !std::isfinite(result.event_dt)
        || result.open_compression_subcells < 1
        || result.open_compression_subcells > 16
        || !(result.open_compression_tolerance > 0.0)
        || !std::isfinite(result.open_compression_tolerance)
        || result.event_dt_convergence_levels < 0
        || result.event_dt_convergence_levels > 8
        || !(result.event_dt_convergence_time >= 0.0)
        || !std::isfinite(result.event_dt_convergence_time)
        || (result.event_dt_convergence_levels > 0
            && !(result.event_dt_convergence_time > 0.0))
        || !(result.potential_fraction > 0.0)
        || !(result.potential_fraction <= 1.0)
        || !std::isfinite(result.potential_fraction)
        || !(result.collision_fraction > 0.0)
        || !(result.collision_fraction <= 1.0)
        || !std::isfinite(result.collision_fraction)
        || !(result.collision_cfl_safety > 0.0)
        || !(result.collision_cfl_safety <= 1.0)
        || !std::isfinite(result.collision_cfl_safety)
        || !(result.maximum_open_mass_fraction >= 0.0)
        || !std::isfinite(result.maximum_open_mass_fraction)
        || !(result.maximum_collision_excluded_open_mass_fraction >= 0.0)
        || !std::isfinite(
            result.maximum_collision_excluded_open_mass_fraction)
        || !(result.maximum_detached_mass_fraction > 0.0)
        || !(result.maximum_detached_mass_fraction <= 1.0)
        || !std::isfinite(result.maximum_detached_mass_fraction)
        || !(result.maximum_unrepresented_mass_fraction >= 0.0)
        || !std::isfinite(result.maximum_unrepresented_mass_fraction)
        || !(result.maximum_collision_escaped_increment_fraction >= 0.0)
        || !std::isfinite(
            result.maximum_collision_escaped_increment_fraction)
        || !(result.maximum_collision_moment_repair_mass_fraction >= 0.0)
        || !std::isfinite(
            result.maximum_collision_moment_repair_mass_fraction)
        || !(result.startup_field_tolerance > 0.0)
        || !std::isfinite(result.startup_field_tolerance)
        || result.startup_field_iterations < 0
        || (result.atlas_initialization != "analytic"
            && result.atlas_initialization != "grid")
        || (result.collision_coupling != "carrier_ntc"
            && result.collision_coupling != "delta_ntc"
            && result.collision_coupling != "owned_ntc"
            && result.collision_coupling != "batched_ntc"
            && result.collision_coupling != "reversible_increment"
            && result.collision_coupling != "detached_ntc_experimental")
        || (result.reversible_operator != "streamed"
            && result.reversible_operator != "sampled")
        || result.sampled_dense_cells < 0
        || result.sampled_min_samples < 1
        || result.sampled_max_samples < result.sampled_min_samples
        || !(result.sampled_guided_fraction >= 0.0)
        || !(result.sampled_guided_fraction < 1.0)
        || !std::isfinite(result.sampled_guided_fraction)
        || !(result.sampled_rhs_tolerance > 0.0)
        || !std::isfinite(result.sampled_rhs_tolerance)
        || !(result.sampled_heat_tolerance > 0.0)
        || !std::isfinite(result.sampled_heat_tolerance)
        || !(result.collision_locality_max_ratio > 0.0)
        || !std::isfinite(result.collision_locality_max_ratio)
        || !(result.collision_cfl > 0.0)
        || !(result.collision_cfl <= 1.0)
        || !std::isfinite(result.collision_cfl)
        || result.threads == 0
        || result.nr < 4 || result.nw < 4 || result.nl < 1
        || !(result.r_min > 0.0) || !(result.r_max > result.r_min)
        || !(result.w_max > 0.0) || !(result.L_max > 0.0)
        || (result.eta_grid != "power"
            && result.eta_grid != "composite_power")
        || !(result.eta_power >= 1.0)
        || (result.eta_grid == "composite_power"
            && (result.eta_inner_cells < 1
                || result.eta_inner_cells >= result.nl
                || !(result.eta_join > 0.0)
                || !(result.eta_join
                    < 0.5 * result.L_max * result.L_max)))) {
        throw std::runtime_error("Invalid canonical-orbit options.");
    }
    return result;
}

struct FrankEddingtonTable {
    std::vector<double> relative_energy;
    std::vector<double> distribution;

    // The table uses positive relative binding energy E=-H, with Phi=0 at
    // infinity.  Interpolate f(E) between neighboring entries and give it zero
    // support outside the tabulated interval.
    double evaluate_hamiltonian(double hamiltonian) const {
        const double energy = -hamiltonian;
        if (relative_energy.size() < 2
            || energy < relative_energy.front()
            || energy > relative_energy.back()) {
            return 0.0;
        }
        const auto upper = std::lower_bound(
            relative_energy.begin(), relative_energy.end(), energy);
        if (upper == relative_energy.begin()) {
            return distribution.front();
        }
        if (upper == relative_energy.end()) {
            return distribution.back();
        }
        const std::size_t right = static_cast<std::size_t>(
            upper - relative_energy.begin());
        const std::size_t left = right - 1;
        const double fraction = (energy - relative_energy[left])
            / (relative_energy[right] - relative_energy[left]);
        return std::max(
            0.0,
            std::fma(
                fraction,
                distribution[right] - distribution[left],
                distribution[left]));
    }
};

// ---- Frank equilibrium import and coarse cell-average state ----------------
FrankEddingtonTable load_frank_eddington_table(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "Could not open Frank Eddington table: " + path.string());
    }
    FrankEddingtonTable table;
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream row(line);
        std::string energy;
        std::string value;
        if (!std::getline(row, energy, ',')
            || !std::getline(row, value)) {
            throw std::runtime_error(
                "Malformed Frank Eddington table row.");
        }
        table.relative_energy.push_back(std::stod(energy));
        table.distribution.push_back(std::max(0.0, std::stod(value)));
    }
    if (table.relative_energy.size() < 2
        || !std::is_sorted(
            table.relative_energy.begin(), table.relative_energy.end())) {
        throw std::runtime_error("Frank Eddington table is incomplete.");
    }
    return table;
}

ev::Params frank_coarse_parameters(
    const RunOptions& run) {
    ev::Params p;
    p.physics_model = "boltzmann_poisson";
    p.vp_universe = "static";
    p.vp_gravity = 1.0;
    p.vp_poisson_source = "total_density";
    p.vp_poisson_reconstruction = "p2_line_mass";
    p.vp_interpolation = "hamiltonian_sldg";
    p.vp_timestep_mode = "frank_dynamical";
    p.vp_dynamical_eta = 0.04;
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = run.eta_grid;
    p.kinetic_eta_power = run.eta_power;
    p.kinetic_eta_join = run.eta_join;
    p.kinetic_eta_inner_cells = run.eta_inner_cells;
    p.Nr = run.nr;
    p.Nw = run.nw;
    p.NL = run.nl;
    p.r_min = run.r_min;
    p.r_max = run.r_max;
    p.radial_grid = "log";
    p.w_max = run.w_max;
    p.L_max = run.L_max;
    p.w_stretch = 0.0;
    p.w_center = 0.0;
    p.inner_boundary = "delayed_return";
    p.outer_boundary = "outflow_vacuum";
    p.momentum_boundary = "outflow_vacuum";
    p.integrator = "semi_lagrangian_kick_drift_kick";
    p.bp_collision_operator = run.reversible_operator == "streamed"
        ? "boltzmann_reversible_streamed"
        : "boltzmann_reversible_sampled";
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = run.collision_strength;
    p.bp_dsmc_total_mass = 1.1252580933331495;
    // Frank quotes sigma/m in units of sigma_s = 4 pi r_s^2/M_s.
    // VENUS uses r_s^2/M_s, so bp_sigma0 = 4 pi (25/9.71).
    p.bp_sigma0 = 32.354198286197665;
    p.bp_boltzmann_scatter_mu_points = 2;
    p.bp_boltzmann_scatter_phi_points = 4;
    p.bp_reversible_candidate_radius = 6;
    p.bp_reversible_max_velocity_cells = std::max(256, p.Nw * p.NL);
    p.bp_reversible_projection_tolerance = 2.0e-10;
    p.bp_reversible_active_shell_floor_fraction = 1.0e-7;
    p.bp_sampled_dense_cells = run.sampled_dense_cells;
    p.bp_sampled_min_samples_per_batch = run.sampled_min_samples;
    p.bp_sampled_max_samples_per_batch = run.sampled_max_samples;
    p.bp_sampled_guided_fraction = run.sampled_guided_fraction;
    p.bp_sampled_rhs_tolerance = run.sampled_rhs_tolerance;
    p.bp_sampled_rhs_floor_fraction = 1.0;
    p.bp_sampled_heat_tolerance = run.sampled_heat_tolerance;
    p.bp_sampled_outgoing_angle_samples = 1;
    p.bp_sampled_ntc_pair_thinning = false;
    p.bp_sampled_tail_fraction = 0.20;
    p.bp_sampled_stratified_fraction = 0.05;
    p.bp_sampled_uniform_fraction = 0.005;
    p.bp_sampled_activity_boost_max = 1.0;
    p.bp_sampled_fail_on_unconverged =
        run.sampled_fail_on_unconverged;
    p.bp_collision_locality_max_ratio = run.collision_locality_max_ratio;
    p.bp_collision_cfl = run.collision_cfl;
    p.restart_from = run.restart.string();
    p.threads = static_cast<int>(run.threads);
    return p;
}

// ---- Canonical/Eulerian consistency and energy diagnostics -----------------
double phase_mass_l1_difference(
    const std::vector<double>& first,
    const std::vector<double>& second,
    const ev::PhaseSpaceCellGeometry& geometry,
    const ev::Params& p) {
    long double difference = 0.0L;
    long double reference = 0.0L;
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t cell = ev::idx(i, j, k, p);
                const double measure = ev::canonical_cell_measure(
                    geometry, i, j, k);
                difference += measure * std::abs(first[cell] - second[cell]);
                reference += measure * std::abs(first[cell]);
            }
        }
    }
    return reference > 0.0L
        ? static_cast<double>(difference / reference) : 0.0;
}

struct CanonicalEnergyDiagnostic {
    double mass = 0.0;
    double kinetic = 0.0;
    double potential = 0.0;

    double total() const {
        return kinetic + potential;
    }
};

CanonicalEnergyDiagnostic canonical_energy_diagnostic(
    const ev::Moments& moments,
    const ev::VlasovPoissonField& field,
    const ev::PhaseSpaceCellGeometry& geometry,
    double hidden_mass) {
    CanonicalEnergyDiagnostic result;
    result.mass = hidden_mass;
    const std::size_t cells = moments.rho.size();
    if (field.phi.size() != cells
        || geometry.radial.edge.size() != cells + 1) {
        throw std::runtime_error(
            "Canonical energy diagnostic size mismatch.");
    }
    for (std::size_t i = 0; i < cells; ++i) {
        // f is cell averaged in the canonical measure dr dw d eta.  Use the
        // matching radial midpoint rule here, exactly as the conservative
        // Poisson solve does, rather than treating rho(r_i) as a spherical
        // shell-volume average on a logarithmic grid.
        const double radial_measure = 4.0 * ev::PI
            * geometry.radial.center[i] * geometry.radial.center[i]
            * geometry.radial.width[i];
        result.mass += radial_measure * moments.rho[i];
        result.kinetic += 0.5 * radial_measure
            * (moments.p[i] + moments.pt[i]);
        result.potential += 0.5 * radial_measure
            * moments.rho[i] * field.phi[i];
    }
    return result;
}

CanonicalEnergyDiagnostic canonical_self_gravitating_energy_diagnostic(
    const ev::Moments& moments,
    const std::vector<double>& radius,
    double time,
    const ev::Params& p,
    const ev::PhaseSpaceCellGeometry& geometry,
    double hidden_mass) {
    const ev::VlasovPoissonField instantaneous_field =
        ev::solve_vlasov_poisson_field(
            moments, radius, time, p, hidden_mass);
    return canonical_energy_diagnostic(
        moments, instantaneous_field, geometry, hidden_mass);
}

void write_canonical_profile(
    std::ostream& output,
    const ev::CanonicalBoltzmannPoissonEventState& state,
    const ev::Moments& moments,
    const ev::PhaseSpaceCellGeometry& geometry) {
    output << std::setprecision(17);
    for (std::size_t i = 0; i < moments.rho.size(); ++i) {
        const double enclosed_mass = i < state.field.source_mass.size()
            ? state.field.source_mass[i] : 0.0;
        output << state.time << ',' << state.step << ','
               << geometry.radial.center[i] << ',' << moments.rho[i] << ','
               << moments.p[i] << ',' << moments.pt[i] << ','
               << moments.j[i] << ',' << enclosed_mass
               << ",0,0,1\n";
    }
    output.flush();
    if (!output) {
        throw std::runtime_error("Failed to flush canonical profiles.csv.");
    }
}

}  // namespace

// ---- Canonical event evolution, checkpointing, and profile output ----------
int main(int argc, char** argv) {
    // ---- Parse controls and reconstruct the coarse Frank equilibrium ------
    const RunOptions run = parse_options(argc, argv);
    ev::MpiContext mpi;
    ev::SolverState state = ev::create_solver_state(
        frank_coarse_parameters(run), mpi);
    ev::initialize_distribution(state);
    const ev::DelayedInnerCarrierLedger carrier =
        ev::evaluate_delayed_inner_carrier_ledger(
            state.inner_carrier, state.kinetic_geometry, state.local_p);
    // Integrate the imported distribution over velocity to obtain the mass
    // density.  Include the mass carried inside the resolved grid when solving
    // Poisson's equation, so the orbit construction sees the full source.
    const ev::Moments initial_moments = ev::compute_global_vp_moments(
        state.f,
        state.r,
        state.w,
        state.L,
        state.wq,
        state.Lq,
        state.local_p,
        state.mpi);
    const ev::VlasovPoissonField field = ev::solve_vlasov_poisson_field(
        initial_moments,
        state.r,
        state.t,
        state.global_p,
        carrier.physical_mass);
    auto potential = [&](double radius) {
        return ev::sample_vp_field(
            radius,
            state.x_r,
            state.r,
            field,
            state.global_p).phi;
    };

    // ---- Load the stationary Eddington f(H) and audit direct grid mass ----
    // H is the single-particle energy in the frozen spherical potential.
    const std::filesystem::path eddington_path = run.eddington_table.empty()
        ? run.restart.parent_path() / "frank_core_eddington_8192.csv"
        : run.eddington_table;
    const FrankEddingtonTable eddington =
        load_frank_eddington_table(eddington_path);
    const auto stationary_density = [eddington](double, double energy) {
        return eddington.evaluate_hamiltonian(energy);
    };
    // Independently integrate f(Phi+w^2/2+eta/r^2) over each (r,w,eta) cell.
    // Four Gauss points along each coordinate resolve the variation within
    // the cell.  The final factor 8*pi^2 is the spherical phase-volume factor
    // after the position and transverse-velocity angles have been integrated.
    const auto& direct_rule = ev::phase_space_gauss_legendre_rule(4);
    long double numerical_field_direct_fh_mass = 0.0L;
    for (int i = 0; i < state.local_p.Nr; ++i) {
        const double r_left = state.kinetic_geometry.radial.edge[
            static_cast<std::size_t>(i)];
        const double r_right = state.kinetic_geometry.radial.edge[
            static_cast<std::size_t>(i + 1)];
        const double r_center = 0.5 * (r_left + r_right);
        const double r_half_width = 0.5 * (r_right - r_left);
        for (int j = 0; j < state.local_p.Nw; ++j) {
            const double w_left = state.kinetic_geometry.radial_velocity.edge[
                static_cast<std::size_t>(j)];
            const double w_right = state.kinetic_geometry.radial_velocity.edge[
                static_cast<std::size_t>(j + 1)];
            const double w_center = 0.5 * (w_left + w_right);
            const double w_half_width = 0.5 * (w_right - w_left);
            for (int k = 0; k < state.local_p.NL; ++k) {
                const double eta_left = state.kinetic_geometry.eta.edge[
                    static_cast<std::size_t>(k)];
                const double eta_right = state.kinetic_geometry.eta.edge[
                    static_cast<std::size_t>(k + 1)];
                const double eta_center = 0.5 * (eta_left + eta_right);
                const double eta_half_width = 0.5 * (eta_right - eta_left);
                long double cell_integral = 0.0L;
                for (std::size_t qr = 0; qr < direct_rule.first.size(); ++qr) {
                    const double radius = std::fma(
                        r_half_width, direct_rule.first[qr], r_center);
                    const double phi = potential(radius);
                    for (std::size_t qw = 0;
                         qw < direct_rule.first.size();
                         ++qw) {
                        const double radial_velocity = std::fma(
                            w_half_width, direct_rule.first[qw], w_center);
                        for (std::size_t qeta = 0;
                             qeta < direct_rule.first.size();
                             ++qeta) {
                            const double eta_value = std::fma(
                                eta_half_width,
                                direct_rule.first[qeta],
                                eta_center);
                            const double energy = phi
                                + 0.5 * radial_velocity * radial_velocity
                                + eta_value / (radius * radius);
                            cell_integral += direct_rule.second[qr]
                                * direct_rule.second[qw]
                                * direct_rule.second[qeta]
                                * stationary_density(eta_value, energy);
                        }
                    }
                }
                numerical_field_direct_fh_mass += cell_integral
                    * r_half_width * w_half_width * eta_half_width;
            }
        }
    }
    numerical_field_direct_fh_mass *=
        8.0L * static_cast<long double>(ev::PI)
        * static_cast<long double>(ev::PI);
    if (run.direct_fh_diagnostic_only) {
        std::cout << std::setprecision(17)
                  << "grid_mass=" << field.matter_mass
                  << " numerical_field_direct_fh_mass="
                  << static_cast<double>(numerical_field_direct_fh_mass)
                  << " relative_fh_mass_deficit="
                  << (field.matter_mass
                      - static_cast<double>(numerical_field_direct_fh_mass))
                      / field.matter_mass
                  << '\n';
        return 0;
    }
    // Compare center samples with the imported grid using the same canonical
    // cell measure.  This separates the error from sampling f at cell centers
    // from the directly integrated mass computed above.
    long double analytic_center_mass = 0.0L;
    long double analytic_center_l1 = 0.0L;
    long double grid_center_l1 = 0.0L;
    for (int i = 0; i < state.local_p.Nr; ++i) {
        const double radius = state.kinetic_geometry.radial.center[
            static_cast<std::size_t>(i)];
        for (int j = 0; j < state.local_p.Nw; ++j) {
            const double velocity =
                state.kinetic_geometry.radial_velocity.center[
                    static_cast<std::size_t>(j)];
            for (int k = 0; k < state.local_p.NL; ++k) {
                const double eta = state.kinetic_geometry.eta.center[
                    static_cast<std::size_t>(k)];
                const std::size_t cell = ev::idx(i, j, k, state.local_p);
                const double energy = potential(radius)
                    + 0.5 * velocity * velocity + eta / (radius * radius);
                const double analytic = stationary_density(eta, energy);
                const double measure = ev::canonical_cell_measure(
                    state.kinetic_geometry, i, j, k);
                analytic_center_mass += measure * analytic;
                analytic_center_l1 += measure
                    * std::abs(analytic - state.f[cell]);
                grid_center_l1 += measure * std::abs(state.f[cell]);
            }
        }
    }
    constexpr long double physical_phase_factor =
        8.0L * static_cast<long double>(ev::PI)
        * static_cast<long double>(ev::PI);

    // ---- Build a new orbit atlas or restore its complete event state ------
    // The atlas covers bound energies H<0.  Its energy and eta quadrature
    // determine the orbit volumes, and its phase cells divide radial motion
    // between turning points.  Weighting the energy partition by sqrt(f)
    // places more rows where the stationary distribution is appreciable.
    ev::CanonicalAtlasBuildOptions atlas_options;
    atlas_options.phase_cells = run.phase_cells;
    atlas_options.mapped_boundary_segments = run.boundary_segments;
    atlas_options.threads = run.threads;
    atlas_options.maximum_patch_depth = 8;
    atlas_options.mapped_volume_tolerance = 5.0e-3;
    atlas_options.maximum_energy = 0.0;
    atlas_options.minimum_energy_guard_fraction =
        run.minimum_energy_guard_fraction;
    atlas_options.energy_action_fraction_power = run.energy_power;
    atlas_options.direct_tensor_quadrature = true;
    atlas_options.mapping_energy_points = run.mapping_energy_points;
    atlas_options.mapping_phase_points = run.mapping_phase_points;
    atlas_options.eta_quadrature_points = run.atlas_eta_points;
    atlas_options.energy_partition_weight = [stationary_density](
            double eta,
            double energy) {
        return std::sqrt(stationary_density(eta, energy));
    };
    atlas_options.stationary_distribution = stationary_density;
    std::optional<ev::CanonicalBoltzmannPoissonEventState> restart_state;
    if (!run.canonical_restart.empty()) {
        restart_state = ev::read_canonical_boltzmann_poisson_restart(
            run.canonical_restart);
    }
    ev::CanonicalAtlasConstructionResult construction;
    double build_seconds = 0.0;
    if (restart_state.has_value()) {
        construction.atlas = restart_state->atlas;
    } else {
        const auto build_begin = std::chrono::steady_clock::now();
        construction = ev::build_canonical_bound_orbit_atlas(
            state.kinetic_geometry,
            state.local_p,
            run.energy_rows,
            potential,
            carrier.physical_mass,
            atlas_options);
        build_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - build_begin).count();
    }
    ev::CanonicalProjectionLedger loaded;
    ev::CanonicalStationaryHiddenCompletion hidden;
    ev::CanonicalStationaryPhaseInitialization stationary;
    // Sum the orbit-cell overlap volumes in each original grid cell.  The
    // uncovered part multiplied by its original f gives the mass outside the
    // atlas representation.  Keep its radial, velocity and eta contributions
    // separately so an inadequate orbit domain can be identified.
    std::vector<long double> atlas_volume_by_eulerian_cell(
        state.f.size(), 0.0L);
    for (const ev::CanonicalOverlap& overlap : construction.atlas.overlap) {
        if (overlap.eulerian_cell < atlas_volume_by_eulerian_cell.size()
            && overlap.volume > 0.0) {
            atlas_volume_by_eulerian_cell[overlap.eulerian_cell] +=
                overlap.volume;
        }
    }
    std::vector<long double> uncovered_mass_by_radius(
        static_cast<std::size_t>(state.local_p.Nr), 0.0L);
    std::vector<long double> uncovered_mass_by_velocity(
        static_cast<std::size_t>(state.local_p.Nw), 0.0L);
    std::vector<long double> uncovered_mass_by_eta(
        static_cast<std::size_t>(state.local_p.NL), 0.0L);
    long double uncovered_mass = 0.0L;
    for (int i = 0; i < state.local_p.Nr; ++i) {
        for (int j = 0; j < state.local_p.Nw; ++j) {
            for (int k = 0; k < state.local_p.NL; ++k) {
                const std::size_t cell = ev::idx(i, j, k, state.local_p);
                const long double cell_volume = ev::canonical_cell_measure(
                    state.kinetic_geometry, i, j, k);
                const long double missing_volume = std::max(
                    0.0L,
                    cell_volume - atlas_volume_by_eulerian_cell[cell]);
                const long double mass = physical_phase_factor
                    * state.f[cell] * missing_volume;
                uncovered_mass += mass;
                uncovered_mass_by_radius[static_cast<std::size_t>(i)] += mass;
                uncovered_mass_by_velocity[static_cast<std::size_t>(j)] += mass;
                uncovered_mass_by_eta[static_cast<std::size_t>(k)] += mass;
            }
        }
    }
    if (!restart_state.has_value() && run.atlas_initialization == "grid") {
        loaded = ev::canonical_orbit_project_from_eulerian(
            construction.atlas, state.f);
        hidden = ev::canonical_orbit_complete_stationary_hidden_phase(
            construction.atlas);
        stationary = ev::canonical_orbit_initialize_stationary_phase(
            construction.atlas);
    } else {
        // The builder has already integrated the supplied Eddington f(H) over
        // each action-energy cell and populated every orbit phase uniformly.
        // Preserve that analytic state instead of routing it through the
        // coarser Eulerian grid before the evolution begins.
        for (const ev::CanonicalOrbitRow& row : construction.atlas.row) {
            for (double mass : row.phase_mass) {
                loaded.source_mass += mass;
            }
            ++stationary.initialized_rows;
        }
        loaded.visible_mass = loaded.source_mass;
        loaded.destination_mass = loaded.source_mass;
        stationary.mass_before = loaded.source_mass;
        stationary.mass_after = loaded.source_mass;
    }

    // Each phase cell stores mass and its first two energy moments.  They
    // determine a quadratic energy reconstruction.  Record how strongly its
    // variation must be reduced to keep the reconstructed distribution
    // nonnegative, while preserving the cell mass.
    double minimum_p2_theta = 1.0;
    std::size_t limited_p2_cells = 0;
    std::size_t populated_p2_cells = 0;
    for (const ev::CanonicalOrbitRow& row : construction.atlas.row) {
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            if (!(row.phase_mass[q] > 0.0)) {
                continue;
            }
            const std::size_t cell = row.first_cell + q;
            const ev::CanonicalEnergyP2Reconstruction mode =
                ev::canonical_energy_p2_reconstruction(
                    row,
                    row.phase_mass[q],
                    row.phase_energy_mass[q],
                    row.phase_energy2_mass[q],
                    construction.atlas.column[cell]);
            ++populated_p2_cells;
            minimum_p2_theta = std::min(
                minimum_p2_theta, mode.positivity_theta);
            limited_p2_cells += mode.positivity_theta < 1.0;
        }
    }

    // ---- Project atlas mass to the Eulerian grid and test frozen transport
    // "Eulerian" denotes the fixed (r,w,eta) grid used for field deposition.
    std::vector<double> represented_initial;
    const ev::CanonicalProjectionLedger represented_initial_ledger =
        ev::canonical_orbit_project_to_eulerian(
            construction.atlas,
            state.kinetic_geometry,
            state.local_p,
            represented_initial);
    const ev::Moments represented_moments = ev::compute_vp_moments(
        represented_initial,
        state.r,
        state.w,
        state.L,
        state.wq,
        state.Lq,
        state.local_p);
    const double represented_hidden_physical_mass =
        static_cast<double>(physical_phase_factor)
        * represented_initial_ledger.hidden_mass;
    const ev::VlasovPoissonField represented_field =
        ev::solve_vlasov_poisson_field(
            represented_moments,
            state.r,
            state.t,
            state.local_p,
            represented_hidden_physical_mass);
    const ev::CanonicalFieldMismatch initial_field_mismatch =
        ev::canonical_field_mismatch(
            field,
            represented_field,
            represented_moments,
            state.kinetic_geometry,
            represented_hidden_physical_mass,
            1.0e-10);

    ev::canonical_orbit_advance_frozen(
        construction.atlas, run.transport_time);
    std::vector<double> transported;
    const ev::CanonicalProjectionLedger transported_ledger =
        ev::canonical_orbit_project_to_eulerian(
            construction.atlas,
            state.kinetic_geometry,
            state.local_p,
            transported);
    const ev::Moments transported_moments = ev::compute_vp_moments(
        transported,
        state.r,
        state.w,
        state.L,
        state.wq,
        state.Lq,
        state.local_p);

    const double frank_core_diagnostic_mass =
        state.local_p.bp_dsmc_total_mass * 250.0 / 1.0e6;
    auto core_average = [&](const ev::Moments& moments) {
        long double mass = 0.0L;
        long double radial_pressure = 0.0L;
        for (int i = 0; i < state.local_p.Nr; ++i) {
            const double left = state.kinetic_geometry.radial.edge[
                static_cast<std::size_t>(i)];
            const double right = state.kinetic_geometry.radial.edge[
                static_cast<std::size_t>(i + 1)];
            const long double shell_volume = 4.0L * ev::PI / 3.0L
                * (right * right * right - left * left * left);
            const long double shell_mass = shell_volume
                * moments.rho[static_cast<std::size_t>(i)];
            const long double shell_pressure = shell_volume
                * moments.p[static_cast<std::size_t>(i)];
            if (mass + shell_mass >= frank_core_diagnostic_mass
                && shell_mass > 0.0L) {
                const long double fraction =
                    (frank_core_diagnostic_mass - mass) / shell_mass;
                const long double radius_cubed = left * left * left
                    + fraction * (right * right * right - left * left * left);
                radial_pressure += fraction * shell_pressure;
                const long double enclosed_volume = 4.0L * ev::PI / 3.0L
                    * radius_cubed;
                return std::array<double, 3>{
                    static_cast<double>(
                        frank_core_diagnostic_mass / enclosed_volume),
                    static_cast<double>(
                        radial_pressure / frank_core_diagnostic_mass),
                    std::cbrt(static_cast<double>(radius_cubed)),
                };
            }
            mass += shell_mass;
            radial_pressure += shell_pressure;
        }
        throw std::runtime_error(
            "Frank central-density diagnostic mass exceeds the represented domain.");
    };
    const std::array<double, 3> grid_core = core_average(initial_moments);
    const std::array<double, 3> loaded_core = core_average(represented_moments);

    std::vector<double> reconstructed = transported;
    double reconstruction_seconds = 0.0;
    double maximum_rebuild_action_shift = 0.0;
    for (int cycle = 0; cycle < run.reconstruction_cycles; ++cycle) {
        ev::canonical_orbit_project_to_eulerian(
            construction.atlas,
            state.kinetic_geometry,
            state.local_p,
            reconstructed);
        ev::CanonicalAtlasBuildOptions rebuild_options = atlas_options;
        rebuild_options.stationary_distribution = {};
        rebuild_options.energy_partition_weight = {};
        rebuild_options.action_edge_by_eta =
            ev::canonical_orbit_action_edges(
                construction.atlas, state.local_p.NL);
        const auto rebuild_begin = std::chrono::steady_clock::now();
        ev::CanonicalAtlasConstructionResult rebuilt =
            ev::build_canonical_bound_orbit_atlas(
                state.kinetic_geometry,
                state.local_p,
                run.energy_rows,
                potential,
                carrier.physical_mass,
                rebuild_options);
        const ev::CanonicalPreservedTopologyTransfer transfer =
            ev::canonical_orbit_transfer_preserved_topology(
                construction.atlas, rebuilt.atlas);
        if (std::abs(transfer.residual)
            > 2.0e-12 * std::max(1.0, transfer.mass_before)) {
            throw std::runtime_error(
                "Preserved-action reconstruction failed its mass ledger.");
        }
        maximum_rebuild_action_shift = std::max(
            maximum_rebuild_action_shift,
            transfer.maximum_normalized_action_edge_shift);
        construction = std::move(rebuilt);
        reconstruction_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - rebuild_begin).count();
    }
    if (run.reconstruction_cycles > 0) {
        ev::canonical_orbit_project_to_eulerian(
            construction.atlas,
            state.kinetic_geometry,
            state.local_p,
            reconstructed);
    }
    const ev::Moments reconstructed_moments = ev::compute_vp_moments(
        reconstructed,
        state.r,
        state.w,
        state.L,
        state.wq,
        state.Lq,
        state.local_p);
    const std::array<double, 3> reconstructed_core = core_average(
        reconstructed_moments);

    constexpr double physical_factor = 8.0 * ev::PI * ev::PI;
    std::cout << std::setprecision(12)
              << "atlas_build_seconds=" << build_seconds
              << " rows=" << construction.atlas.row.size()
              << " columns=" << construction.atlas.column.size()
              << " overlaps=" << construction.atlas.overlap.size()
              << " eta_rows_built="
              << construction.diagnostics.eta_rows_built
              << " eta_rows_open="
              << construction.diagnostics.eta_rows_without_closed_orbits
              << " max_patch_scale_error="
              << construction.diagnostics.maximum_patch_scale_error
              << " quadrature_volume="
              << construction.diagnostics.total_canonical_quadrature_volume
              << " max_column_quadrature_fraction="
              << construction.diagnostics.maximum_column_quadrature_fraction
              << " p2_limited=" << limited_p2_cells
              << '/' << populated_p2_cells
              << " min_p2_theta=" << minimum_p2_theta
              << " eta_quadrature_points=" << run.atlas_eta_points
              << '\n'
              << "grid_mass=" << field.matter_mass
              << " numerical_field_direct_fh_mass="
              << static_cast<double>(numerical_field_direct_fh_mass)
              << " analytic_center_mass="
              << static_cast<double>(
                    physical_phase_factor * analytic_center_mass)
              << " analytic_center_l1="
              << static_cast<double>(analytic_center_l1 / grid_center_l1)
              << " carrier_mass=" << carrier.physical_mass
              << " atlas_initialization=" << run.atlas_initialization
              << " loaded_visible_mass="
              << physical_factor * loaded.destination_mass
              << " completed_hidden_mass="
              << physical_factor * hidden.hidden_mass_added
              << " stationary_mass_residual="
              << stationary.mass_before - stationary.mass_after
              << " represented_mass="
              << physical_factor * represented_initial_ledger.source_mass
              << " represented_mass_residual="
              << represented_initial_ledger.residual
              << " uncovered_grid_mass="
              << static_cast<double>(uncovered_mass) << '\n'
              << "initial_representation_l1="
              << phase_mass_l1_difference(
                    state.f,
                    represented_initial,
                    state.kinetic_geometry,
                    state.local_p)
              << " rho0_grid=" << initial_moments.rho.front()
              << " frozen_transport_l1="
              << phase_mass_l1_difference(
                    represented_initial,
                    transported,
                    state.kinetic_geometry,
                    state.local_p)
              << " rho0_loaded=" << represented_moments.rho.front()
              << " rho0_transported=" << transported_moments.rho.front()
              << " sigma2r0_loaded="
              << represented_moments.p.front()
                    / represented_moments.rho.front()
              << " sigma2r0_grid="
              << initial_moments.p.front() / initial_moments.rho.front()
              << " sigma2r0_transported="
              << transported_moments.p.front()
                    / transported_moments.rho.front()
              << " rho_core_grid=" << grid_core[0]
              << " rho_core_loaded=" << loaded_core[0]
              << " sigma2r_core_grid=" << grid_core[1]
              << " sigma2r_core_loaded=" << loaded_core[1]
              << " transported_mass_residual="
              << transported_ledger.residual
              << " reconstruction_cycles=" << run.reconstruction_cycles
              << " reconstruction_seconds=" << reconstruction_seconds
              << " max_rebuild_action_shift="
              << maximum_rebuild_action_shift
              << " reconstruction_l1=" << phase_mass_l1_difference(
                    represented_initial,
                    reconstructed,
                    state.kinetic_geometry,
                    state.local_p)
              << " rho_core_reconstructed=" << reconstructed_core[0]
              << " sigma2r_core_reconstructed=" << reconstructed_core[1]
              << " initial_phi_mismatch="
              << initial_field_mismatch.potential
              << " initial_phi_mismatch_cell="
              << initial_field_mismatch.potential_cell
              << " initial_force_mismatch="
              << initial_field_mismatch.force
              << " initial_mass_weighted_phi_mismatch="
              << initial_field_mismatch.potential_mass_weighted_rms
              << " initial_mass_weighted_force_mismatch="
              << initial_field_mismatch.force_mass_weighted_rms
              << " initial_force_mismatch_cell="
              << initial_field_mismatch.force_cell
              << " initial_force_mismatch_mass_fraction="
              << initial_field_mismatch.force_cell_enclosed_mass_fraction
              << '\n';
    const auto print_uncovered_support = [&](const char* label,
                                             const std::vector<long double>& mass) {
        std::cout << label << '=';
        bool first = true;
        for (std::size_t q = 0; q < mass.size(); ++q) {
            if (!(mass[q] > 1.0e-10L * std::max(1.0L, uncovered_mass))) {
                continue;
            }
            std::cout << (first ? "" : ";") << q << ':'
                      << static_cast<double>(mass[q]);
            first = false;
        }
        std::cout << '\n';
    };
    print_uncovered_support("uncovered_mass_by_radius", uncovered_mass_by_radius);
    print_uncovered_support("uncovered_mass_by_velocity", uncovered_mass_by_velocity);
    print_uncovered_support("uncovered_mass_by_eta", uncovered_mass_by_eta);

    // ---- Configure local NTC collisions and carrier bookkeeping -----------
    // NTC ("no time counter") samples candidate pairs from a rate majorant.
    // A majorant is an upper bound on the pair encounter rate.  NTC draws
    // trial pairs using that bound and accepts them with the ratio of the
    // actual rate to the bound.  The marker population, radial shell sizes
    // and batch count control how this integral is sampled.
    ev::CanonicalNTCCollisionOptions collision_options;
    collision_options.marker_budget = run.collision_markers;
    collision_options.markers_per_occupied_cell =
        run.collision_markers_per_occupied_cell;
    collision_options.threads = run.threads;
    collision_options.maximum_source_row_geometry_fraction =
        run.source_row_fraction;
    collision_options.maximum_lift_phase_interpolation_cell_error =
        run.maximum_lift_phase_cell_error;
    collision_options.virtual_collision_batches = run.collision_batches;
    collision_options.minimum_expected_trials_per_shell =
        run.minimum_shell_trials;
    collision_options.maximum_virtual_collision_batches =
        run.maximum_collision_batches;
    collision_options.core_importance_radius = run.core_importance_radius;
    collision_options.core_importance_fraction = run.core_importance_fraction;
    collision_options.open_compression.trigger_markers =
        run.open_compression_trigger;
    collision_options.open_compression.phase_subcells =
        run.open_compression_subcells;
    collision_options.open_compression.threads = run.threads;
    collision_options.open_compression.relative_moment_tolerance =
        run.open_compression_tolerance;
    collision_options.collision.target_shell_markers =
        run.collision_target_shell_markers;
    collision_options.collision.minimum_shell_markers =
        run.collision_minimum_shell_markers;
    collision_options.collision.collision_cfl = 1.0;
    collision_options.collision.maximum_substeps = 1;
    collision_options.collision.fail_on_nonlocal_shell = true;
    collision_options.collision.radial_cell_shells = false;
    collision_options.remap.maximum_candidate_radius = 8;
    collision_options.maximum_unrepresented_mass_fraction =
        run.maximum_unrepresented_mass_fraction;
    collision_options.hidden_central_mass = carrier.physical_mass;
    collision_options.retain_collision_products_as_detached_carriers =
        run.collision_coupling == "carrier_ntc"
        || run.collision_coupling == "batched_ntc"
        || run.collision_coupling == "detached_ntc_experimental";
    collision_options.retain_collision_increment_as_signed_carriers =
        run.collision_coupling == "delta_ntc";

    // ---- Initialize scheduled field/collision evolution when requested ---
    if (run.evolution_events > 0 || run.evolution_time > 0.0
        || run.collision_convergence_levels > 0
        || run.collision_marker_convergence_levels > 0
        || run.collision_balanced_convergence_levels > 0
        || run.event_dt_convergence_levels > 0) {
        // Field and collision clocks measure time since their last updates.
        // Fractional-change and encounter bounds determine when each is due.
        // The event routine transports the distribution to the next due time
        // and performs the corresponding updates in the prescribed order.
        ev::CanonicalBoltzmannPoissonEventOptions event_options;
        event_options.energy_rows = run.energy_rows;
        event_options.maximum_collision_interval = run.event_dt;
        event_options.potential_fraction = run.potential_fraction;
        event_options.collision_fraction = run.collision_fraction;
        event_options.collision_cfl_safety = run.collision_cfl_safety;
        event_options.maximum_open_mass_fraction =
            run.maximum_open_mass_fraction;
        event_options.maximum_collision_excluded_open_mass_fraction =
            run.maximum_collision_excluded_open_mass_fraction;
        event_options.maximum_detached_mass_fraction =
            run.maximum_detached_mass_fraction;
        event_options.absorb_detached_carriers_on_field_rebuild =
            run.absorb_detached_carriers_on_field_rebuild;
        event_options.startup_field_tolerance =
            run.startup_field_tolerance;
        event_options.startup_field_iterations =
            run.startup_field_iterations;
        event_options.collision_coupling =
            run.collision_coupling == "owned_ntc"
            ? ev::CanonicalCollisionCoupling::owned_ntc
            : run.collision_coupling == "delta_ntc"
                ? ev::CanonicalCollisionCoupling::delta_ntc
            : run.collision_coupling == "batched_ntc"
                ? ev::CanonicalCollisionCoupling::batched_ntc
            : run.collision_coupling == "reversible_increment"
                ? ev::CanonicalCollisionCoupling::reversible_increment
                : ev::CanonicalCollisionCoupling::carrier_ntc;
        event_options.maximum_collision_escaped_increment_fraction =
            run.maximum_collision_escaped_increment_fraction;
        event_options.maximum_collision_moment_repair_mass_fraction =
            run.maximum_collision_moment_repair_mass_fraction;
        event_options.atlas = atlas_options;
        event_options.collision = collision_options;
        ev::CanonicalBoltzmannPoissonEventState event_state;
        if (restart_state.has_value()) {
            event_state = std::move(*restart_state);
        } else {
            event_state = ev::initialize_canonical_boltzmann_poisson_event_state(
                std::move(construction.atlas),
                field,
                state.t,
                state.step,
                state.kinetic_geometry,
                state.x_r,
                state.r,
                state.w,
                state.L,
                state.wq,
                state.Lq,
                state.local_p,
                event_options);
        }
        const double pristine_time_tolerance = 64.0
            * std::numeric_limits<double>::epsilon();
        if (event_state.step == 0
            && std::abs(event_state.time) <= pristine_time_tolerance
            && event_state.collision.open_marker.empty()
            && event_state.collision.detached_marker.empty()) {
            // A t=0 checkpoint stores the phase-space representation, not a
            // commitment to the event cadence used while creating it.
            event_state.potential_age = 0.0;
            event_state.collision_age = 0.0;
            event_state.collision_timescale =
                event_options.maximum_collision_interval
                / event_options.collision_fraction;
            event_state.collision_majorant_timescale =
                event_options.maximum_collision_interval
                / event_options.collision_cfl_safety;
        }
        // Reproject the restored orbit populations and add every collision
        // carrier.  Their total must reproduce the restart's physical mass
        // before the state is allowed to advance.
        if (!run.canonical_restart.empty()) {
            const std::size_t expected_cells =
                static_cast<std::size_t>(state.local_p.Nr)
                * state.local_p.Nw * state.local_p.NL;
            if (event_state.eulerian_f.size() != expected_cells
                || event_state.field.phi.size()
                    != static_cast<std::size_t>(state.local_p.Nr)) {
                throw std::runtime_error(
                    "Canonical restart does not match the requested grid.");
            }
            std::vector<double> restart_projection;
            const ev::CanonicalProjectionLedger restart_ledger =
                ev::canonical_orbit_project_to_eulerian(
                    event_state.atlas,
                    state.kinetic_geometry,
                    state.local_p,
                    restart_projection);
            const double restart_mass = 8.0 * ev::PI * ev::PI
                * restart_ledger.source_mass
                + ev::canonical_total_carrier_mass(event_state.collision);
            const double restart_mass_scale = std::max(
                1.0, event_state.initial_physical_mass);
            if (std::abs(restart_mass - event_state.initial_physical_mass)
                > 2.0e-10 * restart_mass_scale) {
                throw std::runtime_error(
                    "Canonical restart failed its represented-mass check.");
            }
        }
        // ---- Open append-safe profiles, event history, and mass budgets ---
        std::ofstream history;
        std::ofstream profiles;
        std::ofstream operator_budget;
        double initial_total_energy = 0.0;
        if (!run.output_directory.empty()
            && (run.evolution_events > 0 || run.evolution_time > 0.0)) {
            std::filesystem::create_directories(run.output_directory);
            const bool append_output = !run.canonical_restart.empty()
                && std::filesystem::exists(
                    run.output_directory / "history.csv")
                && std::filesystem::exists(
                    run.output_directory / "profiles.csv")
                && std::filesystem::exists(
                    run.output_directory / "operator_mass_budget.csv");
            history.open(
                run.output_directory / "history.csv",
                append_output ? std::ios::app : std::ios::trunc);
            profiles.open(
                run.output_directory / "profiles.csv",
                append_output ? std::ios::app : std::ios::trunc);
            operator_budget.open(
                run.output_directory / "operator_mass_budget.csv",
                append_output ? std::ios::app : std::ios::trunc);
            if (!history || !profiles || !operator_budget) {
                throw std::runtime_error(
                    "Could not create canonical Frank diagnostic files.");
            }
            if (!append_output) {
              ev::write_canonical_boltzmann_poisson_restart(
                  run.output_directory / "restart_at_start.bin",
                  event_state);
              history
                << "step,time,dt,total_mass,kinetic_energy,potential_energy,"
                   "total_energy,relative_energy_error,core_density,"
                   "core_sigma_r2,core_radius,potential_age,collision_age,"
                   "potential_timescale,collision_timescale,"
                   "collision_majorant_timescale,event_order,"
                   "collision_applied,open_mass_fraction,active_open_mass,"
                   "hidden_open_mass,escaped_open_mass,detached_mass_fraction,"
                   "active_detached_mass,hidden_detached_mass,"
                   "escaped_detached_mass,potential_rebuilt,"
                   "signed_positive_mass,signed_negative_mass,"
                   "signed_positive_markers,signed_negative_markers,"
                   "mass_weighted_potential_change,"
                   "mass_weighted_force_change\n";
              profiles << "time,step,r,rho,p,pt,j,m,mu,lambda,gap\n";
              operator_budget
                << "step,time,dt,transport_mass_before,transport_mass_after,"
                   "transport_mass_delta,"
                   "collision_represented_total_mass_before,"
                   "collision_represented_total_mass_after,"
                   "collision_represented_total_mass_delta,"
                   "collision_mass_before,"
                   "collision_mass_after,collision_mass_delta,"
                   "collision_radial_momentum_delta,"
                   "collision_kinetic_delta,"
                   "collision_mapped_mass_delta,"
                   "collision_mapped_radial_momentum_delta,"
                   "collision_mapped_kinetic_delta,"
                   "atlas_energy_quantization_delta,"
                   "trial_pairs,accepted_pairs,"
                   "skipped_already_collided_trials,"
                   "max_pair_momentum_error,max_pair_kinetic_energy_error,"
                   "atlas_transfer_mass_delta,open_mass,active_open_mass,"
                   "hidden_open_mass,escaped_open_mass,detached_mass,"
                   "active_detached_mass,hidden_detached_mass,"
                   "escaped_detached_mass,absorbed_detached_mass,"
                   "detached_absorption_mass_residual,"
                   "detached_absorption_energy_residual,"
                   "detached_absorption_cached_lifts,"
                   "detached_absorption_exact_lifts,"
                   "unrepresented_eta_mass,"
                   "unrepresented_energy_mass,orbit_failure_mass,"
                   "collision_coupling,collision_seconds,"
                   "sampled_dense_reactions,sampled_complement_reactions,"
                   "sampled_complement_draws,sampled_projection_cache_hits,"
                   "sampled_projection_cache_misses,"
                   "sampled_reused_operator_shells,"
                   "sampled_max_operator_reference_l1,sampled_rhs_error,"
                   "sampled_heat_error,sampled_unconverged_shells,"
                   "sampled_context_cpu_seconds,"
                   "sampled_estimate_cpu_seconds,"
                   "sampled_materialize_cpu_seconds,"
                   "sampled_advance_cpu_seconds,"
                   "sampled_max_shell_seconds,"
                   "collision_max_dr_over_mfp,collision_nonlocal_cells,"
                   "collision_atlas_mass_delta,collision_escaped_increment,"
                   "collision_represented_mass_residual,"
                   "collision_eulerian_increment_l1,"
                   "collision_atlas_increment_l1,"
                   "collision_atlas_positivity_timescale,"
                   "collision_atlas_mass_weighted_positivity_timescale,"
                   "collision_positivity_repaired_mass,"
                   "collision_positivity_repaired_cells,"
                   "collision_positivity_repaired_rows,"
                   "collision_moment_repaired_mass,"
                   "collision_moment_repaired_energy_l1,"
                   "collision_moment_repaired_energy2_l1,"
                   "collision_moment_repaired_cells,"
                   "collision_moment_repaired_rows,"
                   "signed_materialization_mass_residual,"
                   "signed_materialization_hamiltonian_residual,"
                   "signed_materialization_hamiltonian2_residual,"
                   "signed_positive_mass_materialized,"
                   "signed_negative_mass_materialized,"
                   "signed_positive_open_mass_promoted,"
                   "signed_promoted_eta_outside_mass,"
                   "signed_promoted_eta_row_missing_mass,"
                   "signed_promoted_energy_above_mass,"
                   "signed_promoted_energy_gap_mass,"
                   "signed_promoted_orbit_failure_mass,"
                   "signed_positive_mass_retained,"
                   "signed_negative_mass_retained,"
                   "signed_positivity_fallback_mass,"
                   "signed_positivity_fallback_markers,"
                   "signed_cached_phase_lifts,"
                   "signed_exact_phase_lifts,"
                   "open_compression_applied,"
                   "open_compression_active_before,"
                   "open_compression_active_after,"
                   "open_compression_groups,"
                   "open_compression_moment_residual,"
                   "open_compression_seconds\n";
            }
            const ev::Moments output_moments =
                ev::canonical_boltzmann_poisson_event_moments(
                    event_state,
                    state.kinetic_geometry,
                    state.r,
                    state.w,
                    state.L,
                    state.wq,
                    state.Lq,
                    state.local_p);
            const CanonicalEnergyDiagnostic output_energy =
                canonical_self_gravitating_energy_diagnostic(
                    output_moments,
                    state.r,
                    event_state.time,
                    state.local_p,
                    state.kinetic_geometry,
                    event_state.hidden_physical_mass);
            // The reference energy belongs to the beginning of this segment.
            // Rebuild its self-gravitating field from the same full moments
            // used in the output, including unresolved interior mass.
            initial_total_energy = output_energy.total();
            const std::array<double, 3> output_core =
                core_average(output_moments);
            if (!append_output) {
              history << std::setprecision(17)
                    << event_state.step << ',' << event_state.time
                    << ",0," << output_energy.mass << ','
                    << output_energy.kinetic << ',' << output_energy.potential
                    << ',' << output_energy.total()
                    << ",0," << output_core[0] << ',' << output_core[1]
                    << ',' << output_core[2]
                    << ',' << event_state.potential_age << ','
                    << event_state.collision_age << ','
                    << event_state.potential_timescale << ','
                    << event_state.collision_timescale << ','
                    << event_state.collision_majorant_timescale
                    << ",0,0,0,0,0,0,0,0,0,0,0,"
                    << ev::weighted_pic_total_marker_mass(
                        event_state.collision.signed_increment_positive)
                    << ',' << ev::weighted_pic_total_marker_mass(
                        event_state.collision.signed_increment_negative)
                    << ','
                    << event_state.collision.signed_increment_positive.size()
                    << ','
                    << event_state.collision.signed_increment_negative.size()
                    << ",0,0\n";
              history.flush();
              write_canonical_profile(
                  profiles,
                  event_state,
                  output_moments,
                  state.kinetic_geometry);
            }
        }
        std::cout << "startup_field_iterations="
                  << event_state.startup_field_iterations
                  << " startup_field_mismatch="
                  << event_state.startup_field_mismatch << '\n';
        // ---- Collision-batch, marker-count, and cadence convergence probes -
        // Each probe starts from an independent copy of the same event
        // state.  Doubling the batch count measures sampling sensitivity
        // without advancing the state used by the subsequent evolution.
        for (int level = 0;
             level < run.collision_convergence_levels;
             ++level) {
            ev::CanonicalBoltzmannPoissonEventState probe_state = event_state;
            ev::CanonicalBoltzmannPoissonEventOptions probe_options =
                event_options;
            probe_options.collision.virtual_collision_batches =
                run.collision_batches * (1 << level);
            const auto probe_begin = std::chrono::steady_clock::now();
            const ev::CanonicalBoltzmannPoissonEventDiagnostics probe =
                ev::advance_canonical_boltzmann_poisson_event(
                    probe_state,
                    state.kinetic_geometry,
                    state.x_r,
                    state.r,
                    state.w,
                    state.L,
                    state.wq,
                    state.Lq,
                    state.local_p,
                    probe_options);
            const double probe_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - probe_begin).count();
            const ev::Moments probe_moments =
                ev::canonical_boltzmann_poisson_event_moments(
                    probe_state,
                    state.kinetic_geometry,
                    state.r,
                    state.w,
                    state.L,
                    state.wq,
                    state.Lq,
                    state.local_p);
            const std::array<double, 3> probe_core =
                core_average(probe_moments);
            std::cout << "collision_convergence_level=" << level
                      << " batches="
                      << probe_options.collision.virtual_collision_batches
                      << " shell_batches_min="
                      << probe.collision.minimum_shell_virtual_batches
                      << " shell_batches_max="
                      << probe.collision.maximum_shell_virtual_batches
                      << " core_importance_rows="
                      << probe.collision.core_importance_rows
                      << " core_importance_markers="
                      << probe.collision.core_importance_markers
                      << " seconds=" << probe_seconds
                      << " rho_core=" << probe_core[0]
                      << " sigma2r_core=" << probe_core[1]
                      << " trials="
                      << probe.collision.collision.trial_pairs
                      << " accepted="
                      << probe.collision.collision.accepted_pairs
                      << " marker_init_seconds="
                      << probe.collision.marker_initialization_seconds
                      << " pair_seconds="
                      << probe.collision.collision_pair_seconds
                      << " lift_seconds="
                      << probe.collision.outgoing_lift_seconds
                      << " source_reuses="
                      << probe.collision.source_row_geometry_reuses
                      << " source_exact_uses="
                      << probe.collision.source_exact_orbit_uses
                      << " source_exact_builds="
                      << probe.collision.source_exact_orbit_builds
                      << " source_exact_cache_hits="
                      << probe.collision.source_exact_orbit_cache_hits
                      << " source_unique_exact_cells="
                      << probe.collision.source_unique_exact_orbit_cells
                      << " effective_markers="
                      << probe.collision.effective_marker_number
                      << " cached_phase_lifts="
                      << probe.collision.outgoing_cached_phase_lifts
                      << " exact_phase_lifts="
                      << probe.collision.outgoing_exact_phase_lifts
                      << " max_estimated_phase_cell_error="
                      << probe.collision.maximum_estimated_lift_phase_cell_error
                      << " collision_timescale="
                      << probe_state.collision_timescale
                      << " radial_energy_batch_error="
                      << probe.collision.maximum_relative_collision_radial_energy_batch_error
                      << " heat_flux_batch_error="
                      << probe.collision.maximum_relative_collision_heat_flux_batch_error
                      << " open_mass_fraction="
                      << probe.open_mass_fraction
                      << " mass_residual="
                      << probe.collision.physical_mass_residual
                      << '\n';
        }
        // Here only the collision-marker budget doubles.  Compare the core
        // moments and pair conservation with the batch-count probe above.
        for (int level = 0;
             level < run.collision_marker_convergence_levels;
             ++level) {
            ev::CanonicalBoltzmannPoissonEventState probe_state = event_state;
            ev::CanonicalBoltzmannPoissonEventOptions probe_options =
                event_options;
            probe_options.collision.marker_budget =
                run.collision_markers
                * (static_cast<std::size_t>(1) << level);
            const auto probe_begin = std::chrono::steady_clock::now();
            const ev::CanonicalBoltzmannPoissonEventDiagnostics probe =
                ev::advance_canonical_boltzmann_poisson_event(
                    probe_state,
                    state.kinetic_geometry,
                    state.x_r,
                    state.r,
                    state.w,
                    state.L,
                    state.wq,
                    state.Lq,
                    state.local_p,
                    probe_options);
            const double probe_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - probe_begin).count();
            const ev::Moments probe_moments =
                ev::canonical_boltzmann_poisson_event_moments(
                    probe_state,
                    state.kinetic_geometry,
                    state.r,
                    state.w,
                    state.L,
                    state.wq,
                    state.Lq,
                    state.local_p);
            const std::array<double, 3> probe_core =
                core_average(probe_moments);
            std::cout << "collision_marker_convergence_level=" << level
                      << " marker_budget="
                      << probe_options.collision.marker_budget
                      << " seconds=" << probe_seconds
                      << " rho_core=" << probe_core[0]
                      << " sigma2r_core=" << probe_core[1]
                      << " trials="
                      << probe.collision.collision.trial_pairs
                      << " accepted="
                      << probe.collision.collision.accepted_pairs
                      << " saturation_skips="
                      << probe.collision.collision.skipped_already_collided_trials
                      << " minimum_shell_population="
                      << probe.collision.collision.minimum_shell_population
                      << " maximum_shell_width_over_mfp="
                      << probe.collision.collision.maximum_shell_width_over_mean_free_path
                      << " nonlocal_shells="
                      << probe.collision.collision.nonlocal_shells
                      << " radial_energy_batch_error="
                      << probe.collision.maximum_relative_collision_radial_energy_batch_error
                      << " heat_flux_batch_error="
                      << probe.collision.maximum_relative_collision_heat_flux_batch_error
                      << " mass_residual="
                      << probe.collision.physical_mass_residual
                      << '\n';
        }
        // Increase markers and reduce independent batches together.  This
        // examines how a similar sampling effort is distributed between
        // spatial population and repeated estimates of the encounter rate.
        for (int level = 0;
             level < run.collision_balanced_convergence_levels;
             ++level) {
            ev::CanonicalBoltzmannPoissonEventState probe_state = event_state;
            ev::CanonicalBoltzmannPoissonEventOptions probe_options =
                event_options;
            probe_options.collision.marker_budget =
                run.collision_markers
                * (static_cast<std::size_t>(1) << level);
            probe_options.collision.virtual_collision_batches =
                run.collision_batches / (1 << level);
            const auto probe_begin = std::chrono::steady_clock::now();
            const ev::CanonicalBoltzmannPoissonEventDiagnostics probe =
                ev::advance_canonical_boltzmann_poisson_event(
                    probe_state,
                    state.kinetic_geometry,
                    state.x_r,
                    state.r,
                    state.w,
                    state.L,
                    state.wq,
                    state.Lq,
                    state.local_p,
                    probe_options);
            const double probe_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - probe_begin).count();
            const ev::Moments probe_moments =
                ev::canonical_boltzmann_poisson_event_moments(
                    probe_state,
                    state.kinetic_geometry,
                    state.r,
                    state.w,
                    state.L,
                    state.wq,
                    state.Lq,
                    state.local_p);
            const std::array<double, 3> probe_core =
                core_average(probe_moments);
            std::cout << "collision_balanced_convergence_level=" << level
                      << " marker_budget="
                      << probe_options.collision.marker_budget
                      << " batches="
                      << probe_options.collision.virtual_collision_batches
                      << " seconds=" << probe_seconds
                      << " rho_core=" << probe_core[0]
                      << " sigma2r_core=" << probe_core[1]
                      << " trials="
                      << probe.collision.collision.trial_pairs
                      << " accepted="
                      << probe.collision.collision.accepted_pairs
                      << " saturation_skips="
                      << probe.collision.collision.skipped_already_collided_trials
                      << " minimum_shell_population="
                      << probe.collision.collision.minimum_shell_population
                      << " maximum_shell_width_over_mfp="
                      << probe.collision.collision.maximum_shell_width_over_mean_free_path
                      << " radial_energy_batch_error="
                      << probe.collision.maximum_relative_collision_radial_energy_batch_error
                      << " heat_flux_batch_error="
                      << probe.collision.maximum_relative_collision_heat_flux_batch_error
                      << " mass_residual="
                      << probe.collision.physical_mass_residual
                      << '\n';
        }
        // Evolve independent copies to the same physical endpoint with
        // different requested event intervals.  Actual intervals may be
        // shorter because the field and collision bounds remain active.
        for (int level = 0;
             level < run.event_dt_convergence_levels;
             ++level) {
            ev::CanonicalBoltzmannPoissonEventState probe_state = event_state;
            ev::CanonicalBoltzmannPoissonEventOptions probe_options =
                event_options;
            const double requested_dt = run.event_dt * (1 << level);
            probe_options.maximum_collision_interval = requested_dt;
            const double target_time = probe_state.time
                + run.event_dt_convergence_time;
            int probe_events = 0;
            double minimum_dt = std::numeric_limits<double>::infinity();
            double maximum_dt = 0.0;
            double marker_seconds = 0.0;
            double pair_seconds = 0.0;
            double lift_seconds = 0.0;
            double maximum_mass_residual = 0.0;
            const auto probe_begin = std::chrono::steady_clock::now();
            while (probe_state.time < target_time) {
                const double remaining = target_time - probe_state.time;
                const ev::CanonicalBoltzmannPoissonEventDiagnostics probe =
                    ev::advance_canonical_boltzmann_poisson_event(
                        probe_state,
                        state.kinetic_geometry,
                        state.x_r,
                        state.r,
                        state.w,
                        state.L,
                        state.wq,
                        state.Lq,
                        state.local_p,
                        probe_options,
                        remaining);
                ++probe_events;
                minimum_dt = std::min(minimum_dt, probe.dt);
                maximum_dt = std::max(maximum_dt, probe.dt);
                marker_seconds += probe.collision.marker_initialization_seconds;
                pair_seconds += probe.collision.collision_pair_seconds;
                lift_seconds += probe.collision.outgoing_lift_seconds;
                maximum_mass_residual = std::max(
                    maximum_mass_residual,
                    std::abs(probe.collision.physical_mass_residual));
            }
            const double probe_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - probe_begin).count();
            const ev::Moments probe_moments =
                ev::canonical_boltzmann_poisson_event_moments(
                    probe_state,
                    state.kinetic_geometry,
                    state.r,
                    state.w,
                    state.L,
                    state.wq,
                    state.Lq,
                    state.local_p);
            const std::array<double, 3> probe_core =
                core_average(probe_moments);
            std::cout << "event_dt_convergence_level=" << level
                      << " requested_dt=" << requested_dt
                      << " events=" << probe_events
                      << " minimum_dt=" << minimum_dt
                      << " maximum_dt=" << maximum_dt
                      << " seconds=" << probe_seconds
                      << " rho_core=" << probe_core[0]
                      << " sigma2r_core=" << probe_core[1]
                      << " marker_seconds=" << marker_seconds
                      << " pair_seconds=" << pair_seconds
                      << " lift_seconds=" << lift_seconds
                      << " open_mass_fraction="
                      << ev::canonical_open_marker_mass(probe_state.collision)
                            / probe_state.initial_physical_mass
                      << " maximum_mass_residual="
                      << maximum_mass_residual
                      << '\n';
        }
        // ---- Advance the requested physical event sequence ----------------
        const double evolution_target_time = run.evolution_time > 0.0
            ? event_state.time + run.evolution_time
            : std::numeric_limits<double>::infinity();
        const int maximum_evolution_events = run.evolution_events > 0
            ? run.evolution_events
            : (run.evolution_time > 0.0
                ? std::numeric_limits<int>::max() : 0);
        for (int event = 0;
             event < maximum_evolution_events
                 && event_state.time < evolution_target_time;
             ++event) {
            const double remaining_evolution_time =
                evolution_target_time - event_state.time;
            const ev::CanonicalBoltzmannPoissonEventDiagnostics diagnostics =
                ev::advance_canonical_boltzmann_poisson_event(
                    event_state,
                    state.kinetic_geometry,
                    state.x_r,
                    state.r,
                    state.w,
                    state.L,
                    state.wq,
                    state.Lq,
                    state.local_p,
                    event_options,
                    remaining_evolution_time);
            // Compute moments from the complete post-event representation,
            // including matter outside the bound-orbit atlas.  These moments
            // feed both the radial profiles and the independent energy sum.
            const ev::Moments event_moments =
                ev::canonical_boltzmann_poisson_event_moments(
                event_state,
                state.kinetic_geometry,
                state.r,
                state.w,
                state.L,
                state.wq,
                state.Lq,
                state.local_p);
            const std::array<double, 3> event_core = core_average(event_moments);
            const bool reversible_collision =
                event_options.collision_coupling
                == ev::CanonicalCollisionCoupling::reversible_increment;
            const char* collision_coupling_name = reversible_collision
                ? "reversible_increment"
                : event_options.collision_coupling
                        == ev::CanonicalCollisionCoupling::delta_ntc
                    ? "delta_ntc"
                : event_options.collision_coupling
                        == ev::CanonicalCollisionCoupling::owned_ntc
                    ? "owned_ntc"
                    : event_options.collision_coupling
                            == ev::CanonicalCollisionCoupling::batched_ntc
                        ? "batched_ntc"
                    : "carrier_ntc";
            const double collision_mass_before = reversible_collision
                ? diagnostics.reversible_collision.physical_mass_before
                : diagnostics.collision.physical_mass_before;
            const double collision_mass_after = reversible_collision
                ? diagnostics.reversible_collision.physical_mass_after
                : diagnostics.collision.physical_mass_after;
            const double collision_mass_residual = reversible_collision
                ? diagnostics.reversible_collision.physical_mass_residual
                : diagnostics.collision.physical_mass_residual;
            const double collision_momentum_residual = reversible_collision
                ? diagnostics.reversible_collision
                    .physical_radial_momentum_residual
                : diagnostics.collision.physical_radial_momentum_residual;
            const double collision_kinetic_residual = reversible_collision
                ? diagnostics.reversible_collision
                    .physical_kinetic_energy_residual
                : diagnostics.collision.physical_kinetic_energy_residual;
            const double collision_max_dr_over_mfp = reversible_collision
                ? diagnostics.reversible_collision.collision
                    .maximum_radial_cell_width_over_mean_free_path
                : diagnostics.collision.collision
                    .maximum_shell_width_over_mean_free_path;
            const std::size_t collision_nonlocal_cells = reversible_collision
                ? static_cast<std::size_t>(diagnostics.reversible_collision
                    .collision.nonlocal_collision_cells)
                : diagnostics.collision.collision.nonlocal_shells;
            // Separate changes due to transport, pair scattering, projection
            // and carrier transfer.  Their individual residuals identify
            // which operation contributes to a conservation error.
            if (operator_budget) {
                operator_budget << std::setprecision(17)
                    << event_state.step << ',' << event_state.time << ','
                    << diagnostics.dt << ','
                    << diagnostics.transport_mass_before << ','
                    << diagnostics.transport_mass_after << ','
                    << diagnostics.transport_mass_residual << ','
                    << diagnostics.collision_represented_mass_before << ','
                    << diagnostics.collision_represented_mass_after << ','
                    << diagnostics.collision_represented_mass_residual << ','
                    << (diagnostics.collision_applied
                        ? collision_mass_before
                        : diagnostics.transport_mass_after) << ','
                    << (diagnostics.collision_applied
                        ? collision_mass_after
                        : diagnostics.transport_mass_after) << ','
                    << (diagnostics.collision_applied
                        ? collision_mass_residual
                        : 0.0) << ','
                    << collision_momentum_residual << ','
                    << collision_kinetic_residual << ','
                    << diagnostics.reversible_collision
                        .mapped_physical_mass_residual << ','
                    << diagnostics.reversible_collision
                        .mapped_radial_momentum_residual << ','
                    << diagnostics.reversible_collision
                        .mapped_kinetic_energy_residual << ','
                    << diagnostics.collision.atlas_energy_quantization_error
                    << ',' << diagnostics.collision.collision.trial_pairs
                    << ',' << diagnostics.collision.collision.accepted_pairs
                    << ','
                    << diagnostics.collision.collision.skipped_already_collided_trials
                    << ','
                    << diagnostics.collision.collision.max_pair_momentum_error
                    << ','
                    << diagnostics.collision.collision.max_pair_kinetic_energy_error
                    << ',' << diagnostics.transfer.residual << ','
                    << diagnostics.open_mass << ','
                    << diagnostics.active_open_mass << ','
                    << diagnostics.hidden_open_mass << ','
                    << diagnostics.escaped_open_mass << ','
                    << diagnostics.detached_mass << ','
                    << diagnostics.active_detached_mass << ','
                    << diagnostics.hidden_detached_mass << ','
                    << diagnostics.escaped_detached_mass << ','
                    << diagnostics.detached_absorption.absorbed_mass << ','
                    << diagnostics.detached_absorption.physical_mass_residual
                    << ','
                    << diagnostics.detached_absorption.energy_moment_residual
                    << ','
                    << diagnostics.detached_absorption.cached_phase_lifts
                    << ','
                    << diagnostics.detached_absorption.exact_phase_lifts << ','
                    << diagnostics.collision.unrepresented_eta_mass << ','
                    << diagnostics.collision.unrepresented_energy_mass << ','
                    << diagnostics.collision.unrepresented_orbit_geometry_mass
                    << ',' << collision_coupling_name
                    << ',' << diagnostics.collision_seconds
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_dense_reactions
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_complement_reactions
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_complement_draws
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_projection_cache_hits
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_projection_cache_misses
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_reused_operator_shells
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_max_operator_reference_l1
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_max_rhs_relative_error
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_max_heat_relative_error
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_unconverged_shells
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_context_cpu_seconds
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_estimate_cpu_seconds
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_materialize_cpu_seconds
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_advance_cpu_seconds
                    << ',' << diagnostics.reversible_collision.collision
                        .sampled_max_shell_seconds
                    << ',' << collision_max_dr_over_mfp
                    << ',' << collision_nonlocal_cells
                    << ',' << diagnostics.reversible_collision
                        .atlas_mass_residual
                    << ',' << diagnostics.reversible_collision
                        .escaped_increment_mass
                    << ',' << diagnostics.reversible_collision
                        .represented_mass_residual
                    << ',' << diagnostics.reversible_collision
                        .eulerian_increment_l1
                    << ',' << diagnostics.reversible_collision
                        .atlas_increment_l1
                    << ',' << diagnostics.reversible_collision
                        .atlas_positivity_timescale
                    << ',' << diagnostics.reversible_collision
                        .atlas_mass_weighted_positivity_timescale
                    << ',' << diagnostics.reversible_collision
                        .positivity_repaired_mass
                    << ',' << diagnostics.reversible_collision
                        .positivity_repaired_cells
                    << ',' << diagnostics.reversible_collision
                        .positivity_repaired_rows
                    << ',' << diagnostics.reversible_collision
                        .moment_repaired_mass
                    << ',' << diagnostics.reversible_collision
                        .moment_repaired_energy_l1
                    << ',' << diagnostics.reversible_collision
                        .moment_repaired_energy2_l1
                    << ',' << diagnostics.reversible_collision
                        .moment_repaired_cells
                    << ',' << diagnostics.reversible_collision
                        .moment_repaired_rows
                    << ',' << diagnostics.signed_increment_materialization
                        .mass_residual
                    << ',' << diagnostics.signed_increment_materialization
                        .hamiltonian_residual
                    << ',' << diagnostics.signed_increment_materialization
                        .hamiltonian2_residual
                    << ',' << diagnostics.signed_increment_materialization
                        .positive_mass_materialized
                    << ',' << diagnostics.signed_increment_materialization
                        .negative_mass_materialized
                    << ',' << diagnostics.signed_increment_materialization
                        .positive_open_mass_promoted
                    << ',' << diagnostics.signed_increment_materialization
                        .promoted_open_mass_by_reason[1]
                    << ',' << diagnostics.signed_increment_materialization
                        .promoted_open_mass_by_reason[2]
                    << ',' << diagnostics.signed_increment_materialization
                        .promoted_open_mass_by_reason[3]
                    << ',' << diagnostics.signed_increment_materialization
                        .promoted_open_mass_by_reason[4]
                    << ',' << diagnostics.signed_increment_materialization
                        .promoted_open_mass_by_reason[5]
                    << ',' << diagnostics.signed_increment_materialization
                        .positive_mass_retained
                    << ',' << diagnostics.signed_increment_materialization
                        .negative_mass_retained
                    << ',' << diagnostics.signed_increment_materialization
                        .positivity_fallback_mass
                    << ',' << diagnostics.signed_increment_materialization
                        .positivity_fallback_markers
                    << ',' << diagnostics.signed_increment_materialization
                        .cached_phase_lifts
                    << ',' << diagnostics.signed_increment_materialization
                        .exact_phase_lifts
                    << ',' << diagnostics.collision.open_compression.applied
                    << ',' << diagnostics.collision.open_compression.active_before
                    << ',' << diagnostics.collision.open_compression.active_after
                    << ',' << diagnostics.collision.open_compression.groups
                    << ',' << diagnostics.collision.open_compression
                        .maximum_relative_moment_residual
                    << ',' << diagnostics.collision.open_compression
                        .elapsed_seconds
                    << '\n';
                operator_budget.flush();
            }
            if (history.is_open()
                && (event_state.step % run.output_every == 0
                    || event + 1 == maximum_evolution_events
                    || event_state.time >= evolution_target_time)) {
                const CanonicalEnergyDiagnostic output_energy =
                    canonical_self_gravitating_energy_diagnostic(
                        event_moments,
                        state.r,
                        event_state.time,
                        state.local_p,
                        state.kinetic_geometry,
                        event_state.hidden_physical_mass);
                const double relative_energy_error = initial_total_energy != 0.0
                    ? (output_energy.total() - initial_total_energy)
                        / std::abs(initial_total_energy)
                    : 0.0;
                history << std::setprecision(17)
                        << event_state.step << ',' << event_state.time << ','
                        << diagnostics.dt << ',' << output_energy.mass << ','
                        << output_energy.kinetic << ','
                        << output_energy.potential << ','
                        << output_energy.total() << ','
                        << relative_energy_error << ',' << event_core[0] << ','
                        << event_core[1] << ',' << event_core[2] << ','
                        << event_state.potential_age
                        << ',' << event_state.collision_age << ','
                        << event_state.potential_timescale << ','
                        << event_state.collision_timescale << ','
                        << event_state.collision_majorant_timescale << ','
                        << static_cast<int>(diagnostics.plan.order) << ','
                        << (diagnostics.collision_applied ? 1 : 0) << ','
                        << diagnostics.open_mass_fraction << ','
                        << diagnostics.active_open_mass << ','
                        << diagnostics.hidden_open_mass << ','
                        << diagnostics.escaped_open_mass << ','
                        << diagnostics.detached_mass_fraction << ','
                        << diagnostics.active_detached_mass << ','
                        << diagnostics.hidden_detached_mass << ','
                        << diagnostics.escaped_detached_mass << ','
                        << (diagnostics.potential_rebuilt ? 1 : 0) << ','
                        << ev::weighted_pic_total_marker_mass(
                            event_state.collision.signed_increment_positive)
                        << ',' << ev::weighted_pic_total_marker_mass(
                            event_state.collision.signed_increment_negative)
                        << ','
                        << event_state.collision
                            .signed_increment_positive.size()
                        << ','
                        << event_state.collision
                            .signed_increment_negative.size()
                        << ','
                        << diagnostics.mass_weighted_potential_change << ','
                        << diagnostics.mass_weighted_force_change << '\n';
                history.flush();
                write_canonical_profile(
                    profiles,
                    event_state,
                    event_moments,
                    state.kinetic_geometry);
            }
            // A restart records the complete event state at the accepted
            // time, including carriers and both ages.  Its cadence is
            // independent of the smaller diagnostic files.
            if (!run.output_directory.empty()
                && (event_state.step % run.restart_every == 0
                    || event + 1 == maximum_evolution_events
                    || event_state.time >= evolution_target_time)) {
                ev::write_canonical_boltzmann_poisson_restart(
                    run.output_directory / "restart_state.bin",
                    event_state);
            }
            std::cout << "event=" << event + 1
                      << " t=" << event_state.time
                      << " dt=" << diagnostics.dt
                      << " event_order="
                      << static_cast<int>(diagnostics.plan.order)
                      << " collision_applied="
                      << diagnostics.collision_applied
                      << " rho_core=" << event_core[0]
                      << " sigma2r_core=" << event_core[1]
                      << " r_core=" << event_core[2]
                      << " collision_trials="
                      << diagnostics.collision.collision.trial_pairs
                      << " collision_accepted="
                      << diagnostics.collision.collision.accepted_pairs
                      << " collision_saturation_skips="
                      << diagnostics.collision.collision.skipped_already_collided_trials
                      << " collision_batches="
                      << diagnostics.collision.virtual_collision_batches
                      << " collision_shell_batches_min="
                      << diagnostics.collision.minimum_shell_virtual_batches
                      << " collision_shell_batches_max="
                      << diagnostics.collision.maximum_shell_virtual_batches
                      << " collision_core_importance_markers="
                      << diagnostics.collision.core_importance_markers
                      << " collision_mass_residual="
                      << collision_mass_residual
                      << " collision_mapped_mass_residual="
                      << diagnostics.reversible_collision
                            .mapped_physical_mass_residual
                      << " collision_mapped_momentum_residual="
                      << diagnostics.reversible_collision
                            .mapped_radial_momentum_residual
                      << " collision_mapped_kinetic_residual="
                      << diagnostics.reversible_collision
                            .mapped_kinetic_energy_residual
                      << " collision_seconds="
                      << diagnostics.collision_seconds
                      << " signed_materialization_dM="
                      << diagnostics.signed_increment_materialization
                            .mass_residual
                      << " signed_materialization_dH="
                      << diagnostics.signed_increment_materialization
                            .hamiltonian_residual
                      << " signed_materialization_dH2="
                      << diagnostics.signed_increment_materialization
                            .hamiltonian2_residual
                      << " signed_positive_materialized="
                      << diagnostics.signed_increment_materialization
                            .positive_mass_materialized
                      << " signed_negative_materialized="
                      << diagnostics.signed_increment_materialization
                            .negative_mass_materialized
                      << " signed_open_promoted="
                      << diagnostics.signed_increment_materialization
                            .positive_open_mass_promoted
                      << " signed_promoted_eta_outside="
                      << diagnostics.signed_increment_materialization
                            .promoted_open_mass_by_reason[1]
                      << " signed_promoted_eta_row_missing="
                      << diagnostics.signed_increment_materialization
                            .promoted_open_mass_by_reason[2]
                      << " signed_promoted_energy_above="
                      << diagnostics.signed_increment_materialization
                            .promoted_open_mass_by_reason[3]
                      << " signed_promoted_energy_gap="
                      << diagnostics.signed_increment_materialization
                            .promoted_open_mass_by_reason[4]
                      << " signed_promoted_orbit_failure="
                      << diagnostics.signed_increment_materialization
                            .promoted_open_mass_by_reason[5]
                      << " signed_fallback_mass="
                      << diagnostics.signed_increment_materialization
                            .positivity_fallback_mass
                      << " signed_fallback_markers="
                      << diagnostics.signed_increment_materialization
                            .positivity_fallback_markers
                      << " signed_materialization_seconds="
                      << diagnostics.signed_increment_materialization
                            .elapsed_seconds
                      << " signed_positive_compression_before="
                      << diagnostics.signed_increment_materialization
                            .positive_compression.active_before
                      << " signed_positive_compression_after="
                      << diagnostics.signed_increment_materialization
                            .positive_compression.active_after
                      << " signed_positive_compression_retained_groups="
                      << diagnostics.signed_increment_materialization
                            .positive_compression
                            .groups_retained_for_conditioning
                      << " signed_negative_compression_before="
                      << diagnostics.signed_increment_materialization
                            .negative_compression.active_before
                      << " signed_negative_compression_after="
                      << diagnostics.signed_increment_materialization
                            .negative_compression.active_after
                      << " signed_negative_compression_retained_groups="
                      << diagnostics.signed_increment_materialization
                            .negative_compression
                            .groups_retained_for_conditioning
                      << " sampled_dense_reactions="
                      << diagnostics.reversible_collision.collision
                            .sampled_dense_reactions
                      << " sampled_complement_reactions="
                      << diagnostics.reversible_collision.collision
                            .sampled_complement_reactions
                      << " sampled_complement_draws="
                      << diagnostics.reversible_collision.collision
                            .sampled_complement_draws
                      << " sampled_projection_cache_hits="
                      << diagnostics.reversible_collision.collision
                            .sampled_projection_cache_hits
                      << " sampled_projection_cache_misses="
                      << diagnostics.reversible_collision.collision
                            .sampled_projection_cache_misses
                      << " sampled_reused_operator_shells="
                      << diagnostics.reversible_collision.collision
                            .sampled_reused_operator_shells
                      << " sampled_max_operator_reference_l1="
                      << diagnostics.reversible_collision.collision
                            .sampled_max_operator_reference_l1
                      << " sampled_rhs_error="
                      << diagnostics.reversible_collision.collision
                            .sampled_max_rhs_relative_error
                      << " sampled_heat_error="
                      << diagnostics.reversible_collision.collision
                            .sampled_max_heat_relative_error
                      << " sampled_unconverged_shells="
                      << diagnostics.reversible_collision.collision
                            .sampled_unconverged_shells
                      << " sampled_context_cpu_seconds="
                      << diagnostics.reversible_collision.collision
                            .sampled_context_cpu_seconds
                      << " sampled_estimate_cpu_seconds="
                      << diagnostics.reversible_collision.collision
                            .sampled_estimate_cpu_seconds
                      << " sampled_materialize_cpu_seconds="
                      << diagnostics.reversible_collision.collision
                            .sampled_materialize_cpu_seconds
                      << " sampled_advance_cpu_seconds="
                      << diagnostics.reversible_collision.collision
                            .sampled_advance_cpu_seconds
                      << " sampled_max_shell_seconds="
                      << diagnostics.reversible_collision.collision
                            .sampled_max_shell_seconds
                      << " collision_max_dr_over_mfp="
                      << collision_max_dr_over_mfp
                      << " collision_nonlocal_cells="
                      << collision_nonlocal_cells
                      << " collision_atlas_mass_delta="
                      << diagnostics.reversible_collision.atlas_mass_residual
                      << " collision_escaped_increment="
                      << diagnostics.reversible_collision
                            .escaped_increment_mass
                      << " collision_represented_mass_residual="
                      << diagnostics.reversible_collision
                            .represented_mass_residual
                      << " collision_moment_repaired_mass="
                      << diagnostics.reversible_collision
                            .moment_repaired_mass
                      << " collision_moment_repaired_cells="
                      << diagnostics.reversible_collision
                            .moment_repaired_cells
                      << " collision_density_noise="
                      << diagnostics.collision.maximum_relative_density_noise
                      << " collision_heat_flux_noise="
                      << diagnostics.collision.maximum_relative_energy_flux_noise
                      << " collision_radial_energy_batch_error="
                      << diagnostics.collision.maximum_relative_collision_radial_energy_batch_error
                      << " collision_heat_flux_batch_error="
                      << diagnostics.collision.maximum_relative_collision_heat_flux_batch_error
                      << " collision_marker_init_seconds="
                      << diagnostics.collision.marker_initialization_seconds
                      << " collision_marker_layout_seconds="
                      << diagnostics.collision.marker_layout_seconds
                      << " collision_marker_generation_seconds="
                      << diagnostics.collision.marker_generation_seconds
                      << " collision_marker_sort_seconds="
                      << diagnostics.collision.marker_radial_sort_seconds
                      << " collision_markers="
                      << diagnostics.collision.initialization.marker_count
                      << " open_compression_applied="
                      << diagnostics.collision.open_compression.applied
                      << " open_compression_active_before="
                      << diagnostics.collision.open_compression.active_before
                      << " open_compression_active_after="
                      << diagnostics.collision.open_compression.active_after
                      << " open_compression_groups="
                      << diagnostics.collision.open_compression.groups
                      << " open_compression_moment_residual="
                      << diagnostics.collision.open_compression
                            .maximum_relative_moment_residual
                      << " open_compression_seconds="
                      << diagnostics.collision.open_compression.elapsed_seconds
                      << " collision_max_ownership_terms="
                      << diagnostics.collision.maximum_marker_ownership_terms
                      << " collision_pair_seconds="
                      << diagnostics.collision.collision_pair_seconds
                      << " collision_lift_seconds="
                      << diagnostics.collision.outgoing_lift_seconds
                      << " collision_atlas_energy_quantization="
                      << diagnostics.collision.atlas_energy_quantization_error
                      << " collision_positivity_repaired_mass="
                      << diagnostics.reversible_collision
                            .positivity_repaired_mass
                      << " collision_positivity_repaired_cells="
                      << diagnostics.reversible_collision
                            .positivity_repaired_cells
                      << " collision_positivity_repaired_rows="
                      << diagnostics.reversible_collision
                            .positivity_repaired_rows
                      << " collision_atlas_positivity_timescale="
                      << diagnostics.reversible_collision
                            .atlas_positivity_timescale
                      << " collision_atlas_mass_weighted_positivity_timescale="
                      << diagnostics.reversible_collision
                            .atlas_mass_weighted_positivity_timescale
                      << " source_row_geometry_reuses="
                      << diagnostics.collision.source_row_geometry_reuses
                      << " source_exact_uses="
                      << diagnostics.collision.source_exact_orbit_uses
                      << " source_exact_builds="
                      << diagnostics.collision.source_exact_orbit_builds
                      << " source_exact_cache_hits="
                      << diagnostics.collision.source_exact_orbit_cache_hits
                      << " source_effective_markers="
                      << diagnostics.collision.effective_marker_number
                      << " max_source_energy_row_fraction="
                      << diagnostics.collision.maximum_source_energy_row_fraction
                      << " max_source_eta_cell_fraction="
                      << diagnostics.collision.maximum_source_eta_cell_fraction
                      << " cached_phase_lifts="
                      << diagnostics.collision.outgoing_cached_phase_lifts
                      << " exact_phase_lifts="
                      << diagnostics.collision.outgoing_exact_phase_lifts
                      << " max_estimated_phase_cell_error="
                      << diagnostics.collision.maximum_estimated_lift_phase_cell_error
                      << " active_open_collision_sources="
                      << diagnostics.collision.active_open_collision_sources
                      << " active_detached_collision_sources="
                      << diagnostics.collision.active_detached_collision_sources
                      << " canonical_canonical_events="
                      << diagnostics.collision.canonical_canonical_events
                      << " canonical_open_events="
                      << diagnostics.collision.canonical_open_events
                      << " open_open_events="
                      << diagnostics.collision.open_open_events
                      << " open_mass_fraction="
                      << diagnostics.open_mass_fraction
                      << " detached_mass_fraction="
                      << diagnostics.detached_mass_fraction
                      << " high_energy_open_mass="
                      << diagnostics.collision.physical_high_energy_open_mass
                      << " eta_unrepresented_mass="
                      << diagnostics.collision.unrepresented_eta_mass
                      << " missing_eta_row_mass="
                      << diagnostics.collision.unrepresented_missing_eta_row_mass
                      << " energy_unrepresented_mass="
                      << diagnostics.collision.unrepresented_energy_mass
                      << " orbit_failure_mass="
                      << diagnostics.collision.unrepresented_orbit_geometry_mass
                      << " active_open_mass="
                      << diagnostics.active_open_mass
                      << " hidden_open_mass="
                      << diagnostics.hidden_open_mass
                      << " escaped_open_mass="
                      << diagnostics.escaped_open_mass
                      << " active_detached_mass="
                      << diagnostics.active_detached_mass
                      << " hidden_detached_mass="
                      << diagnostics.hidden_detached_mass
                      << " escaped_detached_mass="
                      << diagnostics.escaped_detached_mass
                      << " detached_markers="
                      << event_state.collision.detached_marker.size()
                      << " open_transport_seconds="
                      << diagnostics.open_transport_seconds
                      << " potential_timescale="
                      << event_state.potential_timescale
                      << " collision_timescale="
                      << event_state.collision_timescale
                      << " collision_majorant_timescale="
                      << event_state.collision_majorant_timescale
                      << " potential_mismatch="
                      << diagnostics.relative_potential_mismatch
                      << " force_mismatch="
                      << diagnostics.relative_force_mismatch
                      << " mass_weighted_potential_mismatch="
                      << diagnostics.mass_weighted_potential_mismatch
                      << " mass_weighted_force_mismatch="
                      << diagnostics.mass_weighted_force_mismatch
                      << " mass_weighted_potential_change="
                      << diagnostics.mass_weighted_potential_change
                      << " mass_weighted_force_change="
                      << diagnostics.mass_weighted_force_change
                      << " potential_rebuilt="
                      << diagnostics.potential_rebuilt
                      << " atlas_build_seconds="
                      << diagnostics.atlas_build_seconds
                      << " transfer_mass_residual="
                      << diagnostics.transfer.residual
                      << '\n';
        }
        construction.atlas = std::move(event_state.atlas);
    }

    // ---- Optional collision-only atlas probe ------------------------------
    // This optional probe applies encounters for the requested interval in
    // the frozen potential.  It measures the collision/projection operation
    // separately from the self-consistent sequence of field events above.
    if (run.collision_time > 0.0) {
        ev::CanonicalNTCCollisionState collision_state;
        const auto collision_begin = std::chrono::steady_clock::now();
        const ev::CanonicalNTCCollisionDiagnostics collision =
            ev::apply_canonical_ntc_collision(
                construction.atlas,
                state.kinetic_geometry,
                run.collision_time,
                state.local_p,
                collision_options,
                collision_state,
                potential);
        const double collision_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - collision_begin).count();
        const double collision_majorant_timescale =
            ev::weighted_pic_collision_majorant_timescale(
                collision_state.workspace.shells);
        std::cout << "collision_seconds=" << collision_seconds
                  << " marker_init_seconds="
                  << collision.marker_initialization_seconds
                  << " pair_seconds=" << collision.collision_pair_seconds
                  << " lift_seconds=" << collision.outgoing_lift_seconds
                  << " high_energy_open_mass="
                  << collision.physical_high_energy_open_mass
                  << " eta_unrepresented_mass="
                  << collision.unrepresented_eta_mass
                  << " missing_eta_row_mass="
                  << collision.unrepresented_missing_eta_row_mass
                  << " energy_unrepresented_mass="
                  << collision.unrepresented_energy_mass
                  << " orbit_failure_mass="
                  << collision.unrepresented_orbit_geometry_mass
                  << " source_reuses="
                  << collision.source_row_geometry_reuses
                  << " source_exact_uses="
                  << collision.source_exact_orbit_uses
                  << " source_exact_builds="
                  << collision.source_exact_orbit_builds
                  << " source_exact_cache_hits="
                  << collision.source_exact_orbit_cache_hits
                  << " source_unique_exact_cells="
                  << collision.source_unique_exact_orbit_cells
                  << " effective_markers="
                  << collision.effective_marker_number
                  << " max_source_energy_row_fraction="
                  << collision.maximum_source_energy_row_fraction
                  << " max_source_eta_cell_fraction="
                  << collision.maximum_source_eta_cell_fraction
                  << " markers=" << collision.initialization.marker_count
                  << " collision_batches="
                  << collision.virtual_collision_batches
                  << " shell_batches_min="
                  << collision.minimum_shell_virtual_batches
                  << " shell_batches_max="
                  << collision.maximum_shell_virtual_batches
                  << " core_importance_rows="
                  << collision.core_importance_rows
                  << " core_importance_markers="
                  << collision.core_importance_markers
                  << " majorant_timescale=" << collision_majorant_timescale
                  << " trials=" << collision.collision.trial_pairs
                  << " accepted=" << collision.collision.accepted_pairs
                  << " mass_residual=" << collision.physical_mass_residual
                  << " momentum_residual="
                  << collision.physical_radial_momentum_residual
                  << " energy_residual="
                  << collision.physical_kinetic_energy_residual
                  << " limited_pair_fraction="
                  << collision.positivity_limited_pair_fraction
                  << " unrepresented_mass="
                  << collision.eulerian_to_atlas.escaped_mass
                  << " nonlocal_shells="
                  << collision.collision.nonlocal_shells << '\n';
    }
    return 0;
}
