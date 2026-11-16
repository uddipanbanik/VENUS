// Evolve restart-provided spherical particles with the selected GPU task.
// Inputs are a versioned particle checkpoint and explicit background, mesh,
// time, collision, and output controls. The task fixes the allowed physical
// operator: collisionless VP, calibrated subhalo scattering/capture, or
// elastic material dark matter. This shared driver orders field solves,
// KDK characteristics, collision batches, and publication of completed output.
// Outputs include profiles, event and conservation histories, selected actual
// particle/field snapshots, atomic checkpoints, and a completion/stop status.
// Single-GPU VP/subhalo support static or EdS backgrounds; MPI VP/subhalo use
// EdS, and material tasks use a static halo. MPI calls are collective while
// one root rank publishes files. Carried ages and RNG counters survive a
// restart. Optional endpoint trials restore incoming orbits and fields when
// a newly measured collision deadline requires a shorter step.

#include "venus_gpu/restart.hpp"
#include "venus_gpu/production_driver.hpp"
#include "venus_gpu/production_output.hpp"
#include "venus_gpu/material_sidm.hpp"
#include "venus_gpu/subhalo_collisions.hpp"
#include "venus_gpu/vlasov_poisson.hpp"
#include "ev/sha256.hpp"
#include "venus_gpu/coupled_receiver_clocks.hpp"
#ifdef VENUS_GPU_MPI_RUNNER
#include "venus_gpu/mpi_material_sidm.hpp"
#include "venus_gpu/mpi_subhalo_collisions.hpp"
#include "venus_gpu/mpi_vlasov_poisson.hpp"
#include <mpi.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using venus_gpu::driver::ProductionTask;

// The background determines whether w is physical radial velocity or the
// radial peculiar velocity relative to H*r, and what the weights source.
enum class UniverseMode {
    static_halo,
    eds_expanding,
};

// Task identity selects the collision family before parsing its parameters.
// This keeps material elastic laws separate from subhalo capture/heating.
bool is_material_task(ProductionTask task) {
    return task == ProductionTask::sidm_boltzmann_poisson
        || task == ProductionTask::sidm_constant
        || task == ProductionTask::sidm_power_law
        || task == ProductionTask::sidm_yukawa_born;
}

bool is_explicit_subhalo_task(ProductionTask task) {
    return task == ProductionTask::subhalo_boltzmann_poisson;
}

// Stable task/model names enter metadata and continuation compatibility
// checks, connecting each output directory to its physical interpretation.
const char* task_name(ProductionTask task) {
    switch (task) {
    case ProductionTask::legacy_subhalo_auto:
        return "legacy_subhalo_auto";
    case ProductionTask::vlasov_poisson:
        return "vlasov_poisson";
    case ProductionTask::subhalo_boltzmann_poisson:
        return "subhalo_boltzmann_poisson";
    case ProductionTask::sidm_boltzmann_poisson:
        return "sidm_boltzmann_poisson";
    case ProductionTask::sidm_constant:
        return "sidm_constant";
    case ProductionTask::sidm_power_law:
        return "sidm_power_law";
    case ProductionTask::sidm_yukawa_born:
        return "sidm_yukawa_born";
    }
    throw std::runtime_error("Unknown GPU production task.");
}

const char* cross_section_name(venus_gpu::CrossSectionModel model) {
    switch (model) {
    case venus_gpu::CrossSectionModel::constant: return "constant";
    case venus_gpu::CrossSectionModel::power_law: return "power_law";
    case venus_gpu::CrossSectionModel::yukawa_born: return "yukawa_born";
    }
    throw std::runtime_error("Unknown material-SIDM cross-section model.");
}

// Accept the documented spellings and map them to the same material-law
// enum used by the host and device evaluators.
venus_gpu::CrossSectionModel parse_cross_section_model(
    const std::string& value) {
    if (value == "constant") {
        return venus_gpu::CrossSectionModel::constant;
    }
    if (value == "power_law" || value == "power-law") {
        return venus_gpu::CrossSectionModel::power_law;
    }
    if (value == "yukawa_born" || value == "yukawa-born") {
        return venus_gpu::CrossSectionModel::yukawa_born;
    }
    throw std::runtime_error(
        "Invalid --cross-section-model; expected constant, power_law, or "
        "yukawa_born.");
}

// A focused material executable fixes its law independently of user input.
// The generic material executable instead requires an explicit law choice.
venus_gpu::CrossSectionModel pinned_cross_section(ProductionTask task) {
    switch (task) {
    case ProductionTask::sidm_constant:
        return venus_gpu::CrossSectionModel::constant;
    case ProductionTask::sidm_power_law:
        return venus_gpu::CrossSectionModel::power_law;
    case ProductionTask::sidm_yukawa_born:
        return venus_gpu::CrossSectionModel::yukawa_born;
    default:
        throw std::runtime_error("GPU SIDM task does not pin a model.");
    }
}

// Signal handlers only set this request. Checkpoint I/O occurs later at a
// normal evolution boundary where the particle and clock state can be saved.
std::atomic<bool> stop_requested{false};

#ifdef VENUS_GPU_MPI_RUNNER
int world_rank = 0;
int world_ranks = 1;

void mpi_check(int status, const char* context) {
    if (status == MPI_SUCCESS) {
        return;
    }
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(
        std::string(context) + ": " + std::string(message, length));
}

struct MpiSession {
    MpiSession(int& argc, char**& argv) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_ranks);
    }
    ~MpiSession() {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
    }
};
#endif

bool root_rank() {
#ifdef VENUS_GPU_MPI_RUNNER
    return world_rank == 0;
#else
    return true;
#endif
}

// A request seen by any rank stops the collective calculation at the
// common outer-loop boundary, with all ranks available for checkpoint gather.
bool globally_stop_requested() {
    int stopped = stop_requested.load(std::memory_order_relaxed) ? 1 : 0;
#ifdef VENUS_GPU_MPI_RUNNER
    MPI_Allreduce(
        MPI_IN_PLACE, &stopped, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    return stopped != 0;
}

// Sum disjoint local populations after capture/migration for one physical
// particle-count history, rather than reporting only the root rank's share.
std::size_t global_marker_count(std::size_t local) {
    unsigned long long total = static_cast<unsigned long long>(local);
#ifdef VENUS_GPU_MPI_RUNNER
    MPI_Allreduce(
        MPI_IN_PLACE,
        &total,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        MPI_COMM_WORLD);
#endif
    return static_cast<std::size_t>(total);
}

// Gather complete records for portable checkpoint output. Validate byte
// counts collectively, then restore global ID order on the output rank.
std::vector<venus_gpu::Marker> gather_checkpoint_markers(
    venus_gpu::WpicState& state) {
    std::vector<venus_gpu::Marker> local = state.download();
#ifndef VENUS_GPU_MPI_RUNNER
    return local;
#else
    // MPI byte counts are signed ints. Reject an oversized local payload
    // before any rank enters a gather with an unrepresentable message size.
    int layout_invalid = local.size()
            > static_cast<std::size_t>(INT_MAX) / sizeof(venus_gpu::Marker)
        ? 1 : 0;
    mpi_check(
        MPI_Allreduce(
            MPI_IN_PLACE, &layout_invalid, 1, MPI_INT, MPI_MAX,
            MPI_COMM_WORLD),
        "MPI checkpoint size precondition failed");
    if (layout_invalid != 0) {
        throw std::runtime_error("MPI checkpoint message exceeds INT_MAX.");
    }
    const std::size_t local_bytes_size =
        local.size() * sizeof(venus_gpu::Marker);
    const int local_bytes = static_cast<int>(local_bytes_size);
    std::vector<int> counts(static_cast<std::size_t>(world_ranks), 0);
    mpi_check(
        MPI_Gather(
            &local_bytes, 1, MPI_INT,
            counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD),
        "MPI checkpoint count gather failed");
    std::vector<int> displacements(static_cast<std::size_t>(world_ranks), 0);
    std::uint64_t total_bytes_u64 = 0;
    if (root_rank()) {
        for (int rank = 0; rank < world_ranks; ++rank) {
            displacements[static_cast<std::size_t>(rank)] =
                total_bytes_u64 <= static_cast<std::uint64_t>(INT_MAX)
                ? static_cast<int>(total_bytes_u64) : 0;
            const int bytes = counts[static_cast<std::size_t>(rank)];
            if (bytes < 0 || total_bytes_u64
                    > static_cast<std::uint64_t>(INT_MAX - bytes)) {
                layout_invalid = 1;
                break;
            }
            total_bytes_u64 += static_cast<unsigned int>(bytes);
        }
        if (total_bytes_u64 % sizeof(venus_gpu::Marker) != 0) {
            layout_invalid = 1;
        }
    }
    mpi_check(
        MPI_Bcast(&layout_invalid, 1, MPI_INT, 0, MPI_COMM_WORLD),
        "MPI checkpoint layout broadcast failed");
    if (layout_invalid != 0) {
        throw std::runtime_error(
            "MPI checkpoint layout is invalid or exceeds INT_MAX.");
    }
    const int total_bytes = static_cast<int>(total_bytes_u64);
    std::vector<venus_gpu::Marker> gathered(
        root_rank()
            ? static_cast<std::size_t>(total_bytes) / sizeof(venus_gpu::Marker)
            : 0);
    mpi_check(
        MPI_Gatherv(
            local.empty() ? nullptr : local.data(),
            local_bytes,
            MPI_BYTE,
            gathered.empty() ? nullptr : gathered.data(),
            counts.data(),
            displacements.data(),
            MPI_BYTE,
            0,
            MPI_COMM_WORLD),
        "MPI checkpoint marker gather failed");
    // Rank concatenation depends on ownership. ID sorting gives the
    // portable checkpoint one deterministic global particle sequence.
    if (root_rank()) {
        std::sort(
            gathered.begin(),
            gathered.end(),
            [](const venus_gpu::Marker& first,
               const venus_gpu::Marker& second) {
                return first.id < second.id;
            });
    }
    return gathered;
#endif
}

void request_stop(int) {
    stop_requested.store(true, std::memory_order_relaxed);
}

// Separate physical model controls, global operator cadence, local collision
// CFL controls, and observation cadence. An output time may shorten KDK;
// a collision batch additionally consumes the carried collision age.
struct Options {
    ProductionTask task = ProductionTask::legacy_subhalo_auto;
    bool show_help = false;
    // These optional policies alter field/collision clock selection. Their
    // defaults retain the ordinary task path and require explicit admission.
    bool coupled_receiver_clocks = false;
    bool collision_prestep_coordinate_due = false;
    bool coupled_field_trace = false;
    UniverseMode universe = UniverseMode::eds_expanding;
    fs::path restart;
    fs::path output;
    double end_time = 0.0;
    double maximum_timestep = 0.015;
    double dynamical_fraction = 0.04;
    double output_interval = 10.0;
    std::int64_t history_every = 5000;
    std::int64_t restart_every = 25000;
    std::vector<double> phase_space_snapshot_times;
    std::vector<std::string> invocation;
    int radial_cells = 256;
    double radial_minimum = 0.1;
    double radial_maximum = 1000.0;
    double collision_strength = 0.0;
    // The operator interval governs when a collision batch runs. Inside
    // that batch, collision_cfl and local substeps limit pair probabilities.
    double maximum_collision_interval = 0.5;
    double collision_timescale_fraction = 0.05;
    double collision_cfl = 0.2;
    int collision_maximum_substeps = 16384;
    bool subhalo_adaptive_cell_timestep = false;
    int subhalo_adaptive_batch_substeps = 256;
    int inner_collision_cells = 1;
    int collision_lanes = 8;
    double locality_maximum_ratio = 10.0;
    double velocity_scale = 0.1;
    double plummer_area_per_mass = 1.0;
    double low_speed_regularization = 0.05;

    // Material-SIDM controls.  They use distinct option names so a task cannot
    // accidentally consume a subhalo parameter (or vice versa).
    venus_gpu::CrossSectionModel material_cross_section =
        venus_gpu::CrossSectionModel::constant;
    bool material_cross_section_was_set = false;
    bool subhalo_strength_was_set = false;
    bool material_strength_was_set = false;
    bool saw_subhalo_option = false;
    bool saw_material_option = false;
    bool saw_material_sigma0 = false;
    bool saw_material_power_parameter = false;
    bool saw_material_yukawa_parameter = false;
    double material_collision_strength = 0.0;
    double material_sigma0 = 0.0;
    double material_sigma_power = 4.0;
    double material_sigma_u0 = 0.0;
    double material_yukawa_alpha = 1.0e-2;
    double material_yukawa_mchi = 1.0;
    double material_yukawa_mphi = 1.0;
    std::size_t material_target_shell_markers = 4096;
    std::size_t material_minimum_shell_markers = 32;
    bool material_adaptive_shell_timestep = false;
    bool material_fail_on_nonlocal_shell = false;
};

// An explicit subhalo task requires positive strength; compatibility mode
// can select the collisionless control through its zero-strength convention.
bool subhalo_collisions_enabled(const Options& run) {
    return !is_material_task(run.task) && run.collision_strength > 0.0;
}

bool material_collisions_enabled(const Options& run) {
    return is_material_task(run.task);
}

bool collisions_enabled(const Options& run) {
    return subhalo_collisions_enabled(run)
        || material_collisions_enabled(run);
}

std::string collision_model_name(const Options& run) {
    if (material_collisions_enabled(run)) {
        return std::string("material_sidm_")
            + cross_section_name(run.material_cross_section);
    }
    return subhalo_collisions_enabled(run)
        ? "unequal_mass_plummer_scatter_capture" : "collisionless";
}

// Report each task's allowed model parameters and fixed spatial choices.
// Numerical parsing and task compatibility are checked again below.
void print_usage(const char* executable, ProductionTask task) {
    std::cout
        << "VENUS GPU "
        << (is_material_task(task) ? "static-WPIC " : "expanding-WPIC ")
        << task_name(task)
        << " production driver\n\n"
        << "Usage:\n  " << executable
        << " --restart FILE --output-directory DIR --end-time T [options]\n\n"
        << "Required:\n"
        << "  --restart FILE                         Versioned CPU/GPU WPIC restart; active markers only.\n";
    if (task == ProductionTask::legacy_subhalo_auto
        || is_explicit_subhalo_task(task)) {
        std::cout
        << "                                         Positive subhalo strength also requires consistent internal dispersions.\n";
    }
    std::cout
        << "  --output-directory DIR                Run directory for profiles, histories, and restarts.\n"
        << "  --end-time T                          Final simulation time; T > restart time and T > 0.\n\n"
        << "Evolution and output:\n"
        << "  --universe MODE                       Explicit VP/subhalo: static/eds (single GPU); MPI/compatibility: eds; material SIDM: static.\n"
        << "  --maximum-timestep DT                 Absolute KDK step ceiling; DT > 0 [0.015].\n"
        << "  --dynamical-fraction F                Fraction of the local dynamical time; F > 0 [0.04].\n"
        << "  --output-interval DT                  Profile-time spacing; DT > 0 [10].\n"
        << "  --history-every N                     History cadence in accepted KDK steps; N >= 1 [5000].\n"
        << "  --restart-every N                     Checkpoint cadence in accepted KDK steps; N >= 1 [25000].\n"
        << "  --phase-space-snapshot-times T0,T1,... Complete particles and actual field at selected profile times [off].\n\n"
        << "Logarithmic radial mesh:\n"
        << "  --radial-cells N                      Number of cells; N >= 2 [256].\n"
        << "  --radial-minimum R                    Inner face; R > 0 [0.1].\n"
        << "  --radial-maximum R                    Outer face; R > radial minimum [1000].\n";
    if (task == ProductionTask::legacy_subhalo_auto
        || is_explicit_subhalo_task(task)) {
        std::cout
        << "\nSubhalo collisions:\n"
        << "  --subhalo-collision-strength S        Dimensionless rate multiplier; S >= 0 [0].\n"
        << "  --subhalo-collision-interval DT       Maximum event interval; DT > 0 [0.5].\n"
        << "  --subhalo-collision-timescale-fraction F\n"
        << "                                         Adaptive fraction; 0 <= F <= 1 [0.05].\n"
        << "  --subhalo-collision-cfl C             Collision probability controller; 0 < C <= 1 [0.2].\n"
        << "  --subhalo-collision-maximum-substeps N\n"
        << "                                         Per-event safety limit; N >= 1 [16384].\n"
        << "  --subhalo-adaptive-cell-timestep       Independent cell-local CFL clocks.\n"
        << "  --collision-prestep-coordinate-due     Apply safe coordinate-close deadlines before KDK.\n"
        << "  --subhalo-adaptive-batch-substeps N    Local substeps per CUDA launch; N >= 1 [256].\n"
        << "  --subhalo-inner-collision-cells N     Virtual cells inside the mesh inner face; 1 <= N <= 256 [1].\n"
        << "  --subhalo-collision-lanes N           Deterministic pairing strata; 1 <= N <= 256 [8].\n"
        << "  --subhalo-locality-maximum-ratio X    Stop threshold for max(Delta r/lambda); X >= 0 [10].\n"
        << "  --subhalo-velocity-scale V            Compatibility value; restart dispersions are used; V > 0 [0.1].\n"
        << "  --subhalo-plummer-area-per-mass A     Plummer area per mass; A > 0 [1].\n"
        << "  --subhalo-low-speed-regularization E  Table-floor regularization; 0.05 <= E < 1 [0.05].\n";
    }
    if (is_material_task(task)) {
        std::cout
        << "\nMaterial-SIDM collisions:\n"
        << "  --sidm-collision-strength S           Positive collision-rate multiplier.\n";
        if (task == ProductionTask::sidm_boltzmann_poisson) {
            std::cout
            << "  --cross-section-model MODEL           Required: constant, power_law, or yukawa_born.\n";
        } else {
            std::cout << "  Cross-section model is pinned to "
                      << cross_section_name(pinned_cross_section(task))
                      << ".\n";
        }
        const bool show_constant =
            task == ProductionTask::sidm_boltzmann_poisson
            || pinned_cross_section(task) == venus_gpu::CrossSectionModel::constant;
        const bool show_power_law =
            task == ProductionTask::sidm_boltzmann_poisson
            || pinned_cross_section(task) == venus_gpu::CrossSectionModel::power_law;
        const bool show_yukawa =
            task == ProductionTask::sidm_boltzmann_poisson
            || pinned_cross_section(task) == venus_gpu::CrossSectionModel::yukawa_born;
        if (show_constant || show_power_law) {
            std::cout
            << "  --sidm-sigma0 X                       Constant/power-law normalization; X > 0.\n";
        }
        if (show_power_law) {
            std::cout
            << "  --sidm-sigma-power P                  Power-law exponent; P > 0 [4].\n"
            << "  --sidm-sigma-u0 U                     Power-law softening speed; U > 0.\n";
        }
        if (show_yukawa) {
            std::cout
            << "  --sidm-yukawa-alpha A                 Effective kernel amplitude; A > 0 [0.01].\n"
            << "  --sidm-yukawa-mchi M                  Positive shape parameter [1].\n"
            << "  --sidm-yukawa-mphi M                  Shape parameter; mphi/mchi sets speed [1].\n";
        }
        std::cout
        << "  --sidm-collision-interval DT          Global Boltzmann operator-batch cap; DT > 0 [0.5].\n"
        << "  --sidm-collision-cfl C                Per-substep probability bound; 0 < C <= 1 [0.2].\n"
        << "  --sidm-collision-maximum-substeps N   Per-event safety limit; N >= 1 [16384].\n"
        << "  --sidm-adaptive-shell-timestep        Independent shell-local CFL clocks.\n"
        << "  --sidm-target-shell-markers N         Target radius-local shell occupancy [4096].\n"
        << "  --sidm-minimum-shell-markers N        Minimum collision shell occupancy [32].\n"
        << "  --sidm-fail-on-nonlocal-shell         Stop if a shell exceeds its mean free path.\n";
    }
    std::cout
        << "\n"
        << "Optional receiver field clock:\n"
        << "  --coupled-receiver-clocks             Gravity/H orbit bound, complete receiver RMS field norm, CPU adaptive cadence.\n"
        << "  --coupled-field-trace                 Bounded field/sample caches and observational timing.\n\n"
        << "Information:\n"
        << "  --help, -h                            Print this interface and exit.\n\n"
        << "Fixed spatial physics: G=1, TSC, origin parity, and a reflecting "
           "outer wall.\n";
    if (is_material_task(task)) {
        std::cout
            << "Material SIDM is locked to a static halo; an EdS request is "
               "rejected. Marker masses source the total density.\n";
    } else if (task == ProductionTask::legacy_subhalo_auto) {
        std::cout
            << "The compatibility runner uses EdS expansion "
               "(alpha=2/3, time offset=12). Marker masses represent density contrast.\n";
    } else {
        std::cout
#ifdef VENUS_GPU_MPI_RUNNER
            << "MPI VP/subhalo uses EdS expansion (alpha=2/3, time offset=12). "
               "Marker masses represent density contrast.\n";
#else
            << "VP/subhalo accepts EdS expansion (alpha=2/3, time offset=12) "
               "or a static halo. EdS marker masses represent density contrast. "
               "Static marker masses represent total density.\n";
#endif
    }
    if (task == ProductionTask::legacy_subhalo_auto) {
        std::cout
            << "Compatibility behavior: subhalo strength S=0 selects pure VP; "
               "S>0 selects the subhalo operator.\n";
    } else if (task == ProductionTask::vlasov_poisson) {
        std::cout
            << "This executable rejects every collision option and never "
               "constructs a collision module.\n";
    } else {
        std::cout
            << "This collisional executable requires a positive, explicit "
               "collision strength.\n";
    }
}

// Consume the entire argument and require finiteness, preventing a partially
// parsed value from silently entering a physical parameter or time control.
double parse_double(const char* value, const char* option) {
    std::size_t consumed = 0;
    const std::string text(value);
    const double result = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(result)) {
        throw std::runtime_error(std::string("Invalid ") + option);
    }
    return result;
}

// Integer cadence and indexing inputs retain exact integer interpretation
// before the destination type's range is checked.
long long parse_integer(const char* value, const char* option) {
    std::size_t consumed = 0;
    const std::string text(value);
    const long long result = std::stoll(text, &consumed);
    if (consumed != text.size()) {
        throw std::runtime_error(std::string("Invalid ") + option);
    }
    return result;
}

int parse_int(const char* value, const char* option) {
    const long long result = parse_integer(value, option);
    if (result < static_cast<long long>(INT_MIN)
        || result > static_cast<long long>(INT_MAX)) {
        throw std::runtime_error(std::string("Out-of-range ") + option);
    }
    return static_cast<int>(result);
}

std::size_t parse_size(const char* value, const char* option) {
    const long long result = parse_integer(value, option);
    if (result < 0) {
        throw std::runtime_error(std::string("Negative ") + option);
    }
    return static_cast<std::size_t>(result);
}

// Begin with task-specific defaults, collect explicit user choices, then
// enforce the physical model/background and finite numerical parameter ranges.
Options parse_options(int argc, char** argv, ProductionTask task) {
    Options result;
    result.task = task;
    if (is_material_task(task)) {
        result.universe = UniverseMode::static_halo;
        // Material SIDM uses a fixed global operator-batch cap.  Optional
        // shell-local CFL clocks subcycle independently inside that batch.
        result.collision_timescale_fraction = 0.0;
        if (task != ProductionTask::sidm_boltzmann_poisson) {
            result.material_cross_section = pinned_cross_section(task);
        }
    }
    // Consume a value only when its option expects one; a missing argument
    // is diagnosed before attempting numerical conversion.
    auto need = [&](int& i) {
        if (++i >= argc) {
            throw std::runtime_error(
                std::string("Missing value after ") + argv[i - 1]);
        }
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        // Track which collision family was mentioned even when a supplied
        // strength is zero, so a VP task remains explicitly collisionless.
        if (option.compare(0, 10, "--subhalo-") == 0) {
            result.saw_subhalo_option = true;
        }
        if (option.compare(0, 7, "--sidm-") == 0
            || option == "--cross-section-model") {
            result.saw_material_option = true;
        }
        // State/output and background arguments determine the physical
        // interpretation before the time, mesh, and collision controls follow.
        if (option == "--restart") {
            result.restart = need(i);
        } else if (option == "--output-directory") {
            result.output = need(i);
        } else if (option == "--end-time") {
            result.end_time = parse_double(need(i), "--end-time");
        } else if (option == "--universe") {
            const std::string value(need(i));
            if (value == "static") {
                result.universe = UniverseMode::static_halo;
            } else if (value == "eds" || value == "expanding_eds") {
                result.universe = UniverseMode::eds_expanding;
            } else {
                throw std::runtime_error(
                    "Invalid --universe; expected static or eds.");
            }
        } else if (option == "--collision-prestep-coordinate-due") {
            result.collision_prestep_coordinate_due = true;
        } else if (option == "--coupled-receiver-clocks") {
            result.coupled_receiver_clocks = true;
        } else if (option == "--coupled-field-trace") {
            result.coupled_field_trace = true;
        } else if (option == "--maximum-timestep") {
            result.maximum_timestep = parse_double(
                need(i), "--maximum-timestep");
        } else if (option == "--dynamical-fraction") {
            result.dynamical_fraction = parse_double(
                need(i), "--dynamical-fraction");
        } else if (option == "--output-interval") {
            result.output_interval = parse_double(need(i), "--output-interval");
        } else if (option == "--history-every") {
            result.history_every = parse_integer(need(i), "--history-every");
        } else if (option == "--restart-every") {
            result.restart_every = parse_integer(need(i), "--restart-every");
        } else if (option == "--phase-space-snapshot-times") {
            result.phase_space_snapshot_times =
                venus_gpu::driver::parse_phase_space_snapshot_times(need(i));
        } else if (option == "--radial-cells") {
            result.radial_cells = parse_int(need(i), "--radial-cells");
        } else if (option == "--radial-minimum") {
            result.radial_minimum = parse_double(need(i), "--radial-minimum");
        } else if (option == "--radial-maximum") {
            result.radial_maximum = parse_double(need(i), "--radial-maximum");
        // Subhalo controls govern calibrated scattering/capture and its
        // global and cell-local clocks, including unresolved inner cells.
        } else if (option == "--subhalo-collision-strength") {
            result.collision_strength = parse_double(
                need(i), "--subhalo-collision-strength");
            result.subhalo_strength_was_set = true;
        } else if (option == "--subhalo-collision-interval") {
            result.maximum_collision_interval = parse_double(
                need(i), "--subhalo-collision-interval");
        } else if (option == "--subhalo-collision-timescale-fraction") {
            result.collision_timescale_fraction = parse_double(
                need(i), "--subhalo-collision-timescale-fraction");
        } else if (option == "--subhalo-collision-cfl") {
            result.collision_cfl = parse_double(
                need(i), "--subhalo-collision-cfl");
        } else if (option == "--subhalo-collision-maximum-substeps") {
            result.collision_maximum_substeps = parse_int(
                need(i), "--subhalo-collision-maximum-substeps");
        } else if (option == "--subhalo-adaptive-cell-timestep") {
            result.subhalo_adaptive_cell_timestep = true;
        } else if (option == "--subhalo-collision-adaptive-cell-timestep") {
            result.subhalo_adaptive_cell_timestep =
                parse_int(
                    need(i),
                    "--subhalo-collision-adaptive-cell-timestep") != 0;
        } else if (option == "--subhalo-adaptive-batch-substeps"
                   || option
                       == "--subhalo-collision-adaptive-batch-substeps") {
            result.subhalo_adaptive_batch_substeps = parse_int(
                need(i), "--subhalo-adaptive-batch-substeps");
        } else if (option == "--subhalo-inner-collision-cells") {
            result.inner_collision_cells = parse_int(
                need(i), "--subhalo-inner-collision-cells");
        } else if (option == "--subhalo-collision-lanes") {
            result.collision_lanes = parse_int(
                need(i), "--subhalo-collision-lanes");
        } else if (option == "--subhalo-locality-maximum-ratio") {
            result.locality_maximum_ratio = parse_double(
                need(i), "--subhalo-locality-maximum-ratio");
        } else if (option == "--subhalo-velocity-scale") {
            result.velocity_scale = parse_double(
                need(i), "--subhalo-velocity-scale");
        } else if (option == "--subhalo-plummer-area-per-mass") {
            result.plummer_area_per_mass = parse_double(
                need(i), "--subhalo-plummer-area-per-mass");
        } else if (option == "--subhalo-low-speed-regularization") {
            result.low_speed_regularization = parse_double(
                need(i), "--subhalo-low-speed-regularization");
        // Material controls specify an elastic law and occupancy-based shells.
        // Their option names retain a separate model namespace.
        } else if (option == "--cross-section-model") {
            result.material_cross_section = parse_cross_section_model(need(i));
            result.material_cross_section_was_set = true;
        } else if (option == "--sidm-collision-strength") {
            result.material_collision_strength = parse_double(
                need(i), "--sidm-collision-strength");
            result.material_strength_was_set = true;
        } else if (option == "--sidm-sigma0") {
            result.material_sigma0 = parse_double(need(i), "--sidm-sigma0");
            result.saw_material_sigma0 = true;
        } else if (option == "--sidm-sigma-power") {
            result.material_sigma_power = parse_double(
                need(i), "--sidm-sigma-power");
            result.saw_material_power_parameter = true;
        } else if (option == "--sidm-sigma-u0") {
            result.material_sigma_u0 = parse_double(need(i), "--sidm-sigma-u0");
            result.saw_material_power_parameter = true;
        } else if (option == "--sidm-yukawa-alpha") {
            result.material_yukawa_alpha = parse_double(
                need(i), "--sidm-yukawa-alpha");
            result.saw_material_yukawa_parameter = true;
        } else if (option == "--sidm-yukawa-mchi") {
            result.material_yukawa_mchi = parse_double(
                need(i), "--sidm-yukawa-mchi");
            result.saw_material_yukawa_parameter = true;
        } else if (option == "--sidm-yukawa-mphi") {
            result.material_yukawa_mphi = parse_double(
                need(i), "--sidm-yukawa-mphi");
            result.saw_material_yukawa_parameter = true;
        } else if (option == "--sidm-collision-interval") {
            result.maximum_collision_interval = parse_double(
                need(i), "--sidm-collision-interval");
        } else if (option == "--sidm-collision-cfl") {
            result.collision_cfl = parse_double(need(i), "--sidm-collision-cfl");
        } else if (option == "--sidm-collision-maximum-substeps") {
            result.collision_maximum_substeps = parse_int(
                need(i), "--sidm-collision-maximum-substeps");
        } else if (option == "--sidm-adaptive-shell-timestep") {
            result.material_adaptive_shell_timestep = true;
        } else if (option == "--sidm-target-shell-markers") {
            result.material_target_shell_markers = parse_size(
                need(i), "--sidm-target-shell-markers");
        } else if (option == "--sidm-minimum-shell-markers") {
            result.material_minimum_shell_markers = parse_size(
                need(i), "--sidm-minimum-shell-markers");
        } else if (option == "--sidm-fail-on-nonlocal-shell") {
            result.material_fail_on_nonlocal_shell = true;
        } else if (option == "--help" || option == "-h") {
            result.show_help = true;
        } else {
            throw std::runtime_error("Unknown option: " + option);
        }
    }
    if (result.show_help) {
        return result;
    }
    // Validate model-family intent after parsing. A task accepts only its
    // own collision controls and the explicit strength required by its physics.
    if (task == ProductionTask::vlasov_poisson
        && (result.saw_subhalo_option || result.saw_material_option)) {
        throw std::runtime_error(
            "The Vlasov-Poisson executable rejects all collision options.");
    }
    if (is_explicit_subhalo_task(task)) {
        if (result.saw_material_option) {
            throw std::runtime_error(
                "The subhalo executable rejects material-SIDM options.");
        }
        if (!result.subhalo_strength_was_set
            || !(result.collision_strength > 0.0)) {
            throw std::runtime_error(
                "The subhalo executable requires an explicit positive "
                "--subhalo-collision-strength.");
        }
    }
    if (task == ProductionTask::legacy_subhalo_auto
        && result.saw_material_option) {
        throw std::runtime_error(
            "The compatibility subhalo executable rejects material-SIDM options.");
    }
    if (is_material_task(task)) {
        if (result.saw_subhalo_option) {
            throw std::runtime_error(
                "The material-SIDM executable rejects subhalo options.");
        }
        if (!result.material_strength_was_set
            || !(result.material_collision_strength > 0.0)) {
            throw std::runtime_error(
                "Material SIDM requires an explicit positive "
                "--sidm-collision-strength.");
        }
        if (task == ProductionTask::sidm_boltzmann_poisson
            && !result.material_cross_section_was_set) {
            throw std::runtime_error(
                "The generic material-SIDM executable requires "
                "--cross-section-model.");
        }
        if (task != ProductionTask::sidm_boltzmann_poisson
            && result.material_cross_section_was_set
            && result.material_cross_section != pinned_cross_section(task)) {
            throw std::runtime_error(
                std::string("Pinned SIDM executable requires model ")
                + cross_section_name(pinned_cross_section(task)) + ".");
        }
        // Reject parameters belonging to another material law so the actual
        // rate is determined by an unambiguous set of physical inputs.
        const bool irrelevant_material_parameter =
            (result.material_cross_section
                    == venus_gpu::CrossSectionModel::constant
                && (result.saw_material_power_parameter
                    || result.saw_material_yukawa_parameter))
            || (result.material_cross_section
                    == venus_gpu::CrossSectionModel::power_law
                && result.saw_material_yukawa_parameter)
            || (result.material_cross_section
                    == venus_gpu::CrossSectionModel::yukawa_born
                && (result.saw_material_sigma0
                    || result.saw_material_power_parameter));
        if (irrelevant_material_parameter) {
            throw std::runtime_error(
                "Material-SIDM executable rejects parameters from a different "
                "cross-section model.");
        }
    }
    // Preserve the supported background combinations. The single-GPU
    // explicit Newtonian tasks add static admission alongside EdS.
    const UniverseMode required_universe = is_material_task(task)
        ? UniverseMode::static_halo : UniverseMode::eds_expanding;
#ifdef VENUS_GPU_MPI_RUNNER
    const bool static_newtonian = false;
#else
    const bool static_newtonian = result.universe == UniverseMode::static_halo
        && (task == ProductionTask::vlasov_poisson || is_explicit_subhalo_task(task));
#endif
    if (result.universe != required_universe && !static_newtonian) {
        throw std::runtime_error(is_material_task(task)
            ? "Material-SIDM production is locked to --universe static."
            : "Selected GPU task rejects this universe.");
    }
    // The pre-step due check and trace depend on the receiver policy. That
    // policy requires a fresh single-GPU adaptive-cell subhalo segment.
    if (result.collision_prestep_coordinate_due && !result.coupled_receiver_clocks)
        throw std::runtime_error("Pre-step coordinate due decisions require coupled receiver clocks.");
    if (result.coupled_field_trace && !result.coupled_receiver_clocks)
        throw std::runtime_error("Field tracing requires coupled receiver clocks.");
    if (result.coupled_receiver_clocks) {
#ifdef VENUS_GPU_MPI_RUNNER
        throw std::runtime_error("Coupled receiver clocks require one GPU.");
#endif
        if (!is_explicit_subhalo_task(task) || !result.subhalo_adaptive_cell_timestep
            || !(result.collision_timescale_fraction > 0))
            throw std::runtime_error("Coupled receiver clocks require explicit adaptive-cell subhalo BP.");
        if (fs::exists(result.output) && !fs::is_empty(result.output))
            throw std::runtime_error("Coupled receiver clocks require a fresh output segment.");
    }
    // Validate positive scales, finite work bounds, physical mesh ordering,
    // and local probability fractions before any device state is created.
    if (result.restart.empty() || result.output.empty()
        || !(result.end_time > 0.0) || !(result.maximum_timestep > 0.0)
        || !(result.dynamical_fraction > 0.0)
        || !(result.output_interval > 0.0) || result.history_every < 1
        || result.restart_every < 1 || result.radial_cells < 2
        || !(result.radial_minimum > 0.0)
        || !(result.radial_maximum > result.radial_minimum)
        || !(result.collision_strength >= 0.0)
        || !(result.maximum_collision_interval > 0.0)
        || !(result.collision_timescale_fraction >= 0.0)
        || !(result.collision_timescale_fraction <= 1.0)
        || !(result.collision_cfl > 0.0)
        || !(result.collision_cfl <= 1.0)
        || result.collision_maximum_substeps < 1
        || result.subhalo_adaptive_batch_substeps < 1
        || result.inner_collision_cells < 1
        || result.inner_collision_cells > 256
        || result.collision_lanes < 1 || result.collision_lanes > 256
        || !(result.locality_maximum_ratio >= 0.0)
        || !(result.velocity_scale > 0.0)
        || !(result.plummer_area_per_mass > 0.0)
        || !(result.low_speed_regularization >= 0.05)
        || !(result.low_speed_regularization < 1.0)) {
        throw std::runtime_error("Invalid GPU production controls.");
    }
    if (is_material_task(task)) {
        const bool material_model_valid =
            (result.material_cross_section == venus_gpu::CrossSectionModel::constant
                && result.material_sigma0 > 0.0)
            || (result.material_cross_section
                    == venus_gpu::CrossSectionModel::power_law
                && result.material_sigma0 > 0.0
                && result.material_sigma_power > 0.0
                && result.material_sigma_u0 > 0.0)
            || (result.material_cross_section
                    == venus_gpu::CrossSectionModel::yukawa_born
                && result.material_yukawa_alpha > 0.0
                && result.material_yukawa_mchi > 0.0
                && result.material_yukawa_mphi > 0.0);
        if (!material_model_valid
            || result.material_target_shell_markers < 2
            || result.material_minimum_shell_markers < 2
            || result.material_minimum_shell_markers
                > result.material_target_shell_markers) {
            throw std::runtime_error(
                "Invalid material-SIDM cross section or shell controls.");
        }
    }
    venus_gpu::driver::validate_phase_space_snapshot_times(
        result.phase_space_snapshot_times, result.output_interval);
    for (int i = 0; i < argc; ++i) result.invocation.emplace_back(argv[i]);
    return result;
}

// Construct a logarithmic radial mesh and set the two prescribed faces
// exactly, giving CPU/GPU sidecars the same endpoint geometry.
std::vector<double> logarithmic_edges(const Options& run) {
    std::vector<double> edge(static_cast<std::size_t>(run.radial_cells) + 1);
    const double low = std::log(run.radial_minimum);
    const double high = std::log(run.radial_maximum);
    for (int i = 0; i <= run.radial_cells; ++i) {
        edge[static_cast<std::size_t>(i)] = std::exp(
            low + static_cast<double>(i) / run.radial_cells * (high - low));
    }
    // Match the VENUS CPU cell-face constructor at the boundaries.
    edge.front() = run.radial_minimum;
    edge.back() = run.radial_maximum;
    return edge;
}

// Use the production TSC deposition, regular-origin parity, and reflecting
// outer wall. The universe selects expansion; input weights supply its source.
venus_gpu::VlasovPoissonOptions vp_options(const Options& run) {
    venus_gpu::VlasovPoissonOptions options;
    options.gravity = 1.0;
    options.particle_shape = venus_gpu::ParticleShape::tsc;
    options.inner_boundary = venus_gpu::InnerBoundary::origin_parity;
    options.outer_boundary = venus_gpu::OuterBoundary::reflecting_wall;
    options.expansion.enabled = run.universe == UniverseMode::eds_expanding;
    options.expansion.alpha = 2.0 / 3.0;
    options.expansion.time_offset = 12.0;
    // The restart masses already represent the task's Poisson source:
    // density contrast for EdS VP/subhalo runs and total density for static
    // VP/subhalo/material-SIDM runs. Neither contract subtracts rho_bar inside this
    // module.
    options.subtract_homogeneous_background = false;
    options.carry_active_mass_inside_inner_face = true;
    return options;
}

// Pass encounter controls through to the subhalo operator. Its actual
// pair speed scale comes from the restart's internal dispersions.
venus_gpu::SubhaloCollisionOptions collision_options(const Options& run) {
    venus_gpu::SubhaloCollisionOptions options;
    options.velocity_scale = run.velocity_scale;
    options.plummer_area_per_mass = run.plummer_area_per_mass;
    options.low_speed_regularization = run.low_speed_regularization;
    options.strength = run.collision_strength;
    options.collision_cfl = run.collision_cfl;
    options.maximum_substeps = run.collision_maximum_substeps;
    options.adaptive_cell_timestep =
        run.subhalo_adaptive_cell_timestep;
    options.adaptive_batch_substeps =
        run.subhalo_adaptive_batch_substeps;
    options.inner_collision_cells = run.inner_collision_cells;
    options.collision_lanes = run.collision_lanes;
    options.minimum_cell_population = 2;
    return options;
}

// Pass the selected material law and shell-local sampling controls without
// changing the physical cross-section parameters between host and device.
venus_gpu::CollisionOptions material_collision_options(const Options& run) {
    venus_gpu::CollisionOptions options;
    options.cross_section = run.material_cross_section;
    options.collision_strength = run.material_collision_strength;
    options.sigma0 = run.material_sigma0;
    options.sigma_power = run.material_sigma_power;
    options.sigma_u0 = run.material_sigma_u0;
    options.yukawa_alpha = run.material_yukawa_alpha;
    options.yukawa_mchi = run.material_yukawa_mchi;
    options.yukawa_mphi = run.material_yukawa_mphi;
    options.target_shell_markers = run.material_target_shell_markers;
    options.minimum_shell_markers = run.material_minimum_shell_markers;
    options.collision_cfl = run.collision_cfl;
    options.maximum_substeps = run.collision_maximum_substeps;
    options.adaptive_shell_timestep =
        run.material_adaptive_shell_timestep;
    options.fail_on_nonlocal_shell = run.material_fail_on_nonlocal_shell;
    return options;
}

// Select the orbit step ceiling before clipping to output/end/collision
// times. The optional receiver policy uses its own gravitational bound.
double choose_timestep(
    const venus_gpu::FieldHost& field,
    const venus_gpu::RadialMesh& mesh,
    double time,
    const Options& run) {
    // Collisionless WPIC uses a fixed KDK step (except exact output/final
    // landings), which preserves the time-symmetric bounded energy error of a
    // constant-potential control.  Collision-bearing tasks retain the local
    // dynamical limiter around their independently scheduled operators.
    if (!collisions_enabled(run)) {
        return run.maximum_timestep;
    }
    if (run.coupled_receiver_clocks) {
        return venus_gpu_receiver::gravitational_timestep(
            field, mesh, time, run.maximum_timestep, run.dynamical_fraction);
    }
    double timestep = run.maximum_timestep;
    const double deposited = field.diagnostics.deposited_mesh_mass;
    // Only resolved cells above the source-mass floor set this orbit bound.
    // All particles still source the field and undergo characteristic motion.
    const double floor = 1.0e-12 * deposited;
    const double hubble = run.universe == UniverseMode::eds_expanding
        ? (2.0 / 3.0) / (time + 12.0) : 0.0;
    for (std::size_t cell = 0; cell < field.shell_mass.size(); ++cell) {
        const double mass = field.shell_mass[cell];
        if (!(mass > floor)) {
            continue;
        }
        const double radius = mesh.radius_center[cell];
        const double tangential =
            field.shell_tangential_second_moment[cell] / mass;
        // Gravity contributes abs(dPhi/dr)/r, tangential motion contributes
        // v_t^2/r^2, and expansion adds H^2 to the resolved frequency estimate.
        const double omega2 = std::abs(field.phi_r[cell]) / radius
            + tangential / (radius * radius) + hubble * hubble;
        if (omega2 > 0.0 && std::isfinite(omega2)) {
            timestep = std::min(
                timestep, run.dynamical_fraction / std::sqrt(omega2));
        }
    }
    if (!(timestep > 0.0) || !std::isfinite(timestep)) {
        throw std::runtime_error("GPU production selected invalid timestep.");
    }
    return timestep;
}

// Estimate a cell gravitational time from enclosed source mass in G=1
// units. The hidden interior source enters before the outward mass sum.
double minimum_dynamical_timescale(
    const venus_gpu::FieldHost& field,
    const venus_gpu::RadialMesh& mesh) {
    long double enclosed = field.diagnostics.hidden_inner_mass;
    const double floor = 1.0e-14
        * std::max(1.0, field.diagnostics.deposited_mesh_mass);
    double result = std::numeric_limits<double>::infinity();
    for (std::size_t cell = 0; cell < field.shell_mass.size(); ++cell) {
        enclosed += field.shell_mass[cell];
        const long double radius = mesh.radius_center[cell];
        if (!(enclosed > floor)) {
            continue;
        }
        const long double frequency2 = enclosed / (radius * radius * radius);
        if (frequency2 > 0.0L) {
            result = std::min(
                result, 1.0 / std::sqrt(static_cast<double>(frequency2)));
        }
    }
    return result;
}

// Estimate a maximum-norm field-variation time from two observations.
// Combine deposited mass and effective counts to remove the existing
// sampling floor in quadrature before interpreting a resolved change.
double minimum_potential_change_timescale(
    const venus_gpu::FieldHost& previous,
    const venus_gpu::FieldHost& current,
    double dt) {
    if (!(dt > 0.0) || previous.phi.size() != current.phi.size()) {
        return std::numeric_limits<double>::infinity();
    }
    long double mass = 0.0L;
    long double squared_mass = 0.0L;
    for (std::size_t cell = 0; cell < current.shell_mass.size(); ++cell) {
        mass += current.shell_mass[cell];
        const double neff = current.effective_marker_number[cell];
        if (current.shell_mass[cell] > 0.0 && neff > 0.0) {
            squared_mass += static_cast<long double>(
                current.shell_mass[cell]) * current.shell_mass[cell] / neff;
        }
    }
    if (!(mass > 0.0L) || !(squared_mass > 0.0L)) {
        return std::numeric_limits<double>::infinity();
    }
    const double neff = static_cast<double>(mass * mass / squared_mass);
    // Potential and radial force have separate scales. Compare their
    // largest cell differences to the larger endpoint norm in each channel.
    double phi_norm = 0.0;
    double force_norm = 0.0;
    double phi_difference = 0.0;
    double force_difference = 0.0;
    for (std::size_t cell = 0; cell < current.phi.size(); ++cell) {
        phi_norm = std::max({phi_norm, std::abs(previous.phi[cell]),
            std::abs(current.phi[cell])});
        force_norm = std::max({force_norm, std::abs(previous.phi_r[cell]),
            std::abs(current.phi_r[cell])});
        phi_difference = std::max(phi_difference,
            std::abs(current.phi[cell] - previous.phi[cell]));
        force_difference = std::max(force_difference,
            std::abs(current.phi_r[cell] - previous.phi_r[cell]));
    }
    const double noise = std::max(
        4096.0 * std::numeric_limits<double>::epsilon(),
        0.25 / std::sqrt(neff));
    auto clock = [&](double difference, double norm) {
        const double relative = norm > 0.0 ? difference / norm : 0.0;
        if (!(relative > noise)) {
            return std::numeric_limits<double>::infinity();
        }
        const double resolved = std::sqrt(
            std::max(0.0, relative * relative - noise * noise));
        return resolved > 0.0
            ? dt / resolved : std::numeric_limits<double>::infinity();
    };
    return std::min(
        clock(phi_difference, phi_norm),
        clock(force_difference, force_norm));
}

// Select a global batch interval from its configured cap and active
// physical timescales. Adaptive subhalo cells enforce encounter exposure
// locally, so the coupled policy omits that local rate from the global bound.
double selected_collision_interval(
    const Options& run,
    double collision_time,
    double dynamical_time,
    double potential_time) {
    if (run.coupled_receiver_clocks) {
        return ev::weighted_pic_collision_selection_interval(
            run.maximum_collision_interval, run.collision_timescale_fraction,
            run.subhalo_adaptive_cell_timestep
                ? std::numeric_limits<double>::infinity() : collision_time,
            dynamical_time, potential_time);
    }
    if (run.collision_timescale_fraction == 0.0) {
        return run.maximum_collision_interval;
    }
    const double controlling = std::min(
        {collision_time, dynamical_time, potential_time});
    return std::isfinite(controlling)
        ? std::min(run.maximum_collision_interval,
              run.collision_timescale_fraction * controlling)
        : run.maximum_collision_interval;
}

// This roundoff allowance compares elapsed collision age with its selected
// deadline. It remains distinct from time-coordinate resolution in the loop.
double deadline_tolerance(double age, double deadline) {
    return 64.0 * std::numeric_limits<double>::epsilon()
        * std::max({1.0, std::abs(age), std::abs(deadline)});
}

bool deadline_safe(double age, double deadline) {
    return age <= deadline + deadline_tolerance(age, deadline);
}

void write_profile(
    const fs::path& directory,
    std::int64_t index,
    double time,
    const venus_gpu::FieldHost& field,
    const venus_gpu::RadialMesh& mesh) {
    venus_gpu::driver::write_profile_csv(
        directory.string(), index, time, field, mesh);
}

// Publish the radial geometry used to interpret a restart and its profiles
// through one flushed temporary-file replacement.
void write_mesh_contract(const fs::path& directory, const Options& run) {
    const fs::path temporary = directory / "radial_mesh_contract.txt.tmp";
    std::ofstream output(temporary);
    output << std::setprecision(17)
           << "radial_grid=log\nradial_cells=" << run.radial_cells
           << "\nradial_minimum=" << run.radial_minimum
           << "\nradial_maximum=" << run.radial_maximum << '\n';
    output.flush();
    if (!output) {
        throw std::runtime_error("Failed writing GPU radial-mesh contract.");
    }
    output.close();
    fs::rename(temporary, directory / "radial_mesh_contract.txt");
}

// Escape literal strings before writing paths, task names, or invocation
// arguments into machine-readable output metadata.
std::string json_string(const std::string& value) {
    std::ostringstream escaped;
    escaped << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"': escaped << "\\\""; break;
        case '\\': escaped << "\\\\"; break;
        case '\b': escaped << "\\b"; break;
        case '\f': escaped << "\\f"; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default:
            if (character < 0x20) {
                escaped << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0')
                        << static_cast<unsigned int>(character)
                        << std::dec << std::setfill(' ');
            } else {
                escaped << static_cast<char>(character);
            }
        }
    }
    escaped << '"';
    return escaped.str();
}

// Record the physical task, universe/source convention, model parameters,
// and numerical clock choices alongside the initial population and time.
void write_metadata(
    const fs::path& directory,
    const Options& run,
    const venus_gpu::RestartState& restart) {
    const fs::path temporary = directory / "metadata.json.tmp";
    std::ofstream output(temporary);
    output << std::setprecision(17)
           << "{\n  \"code\": \"VENUS_GPU\",\n"
           << "  \"source_restart\": "
           << json_string(run.restart.string()) << ",\n"
           << "  \"start_time\": " << restart.time << ",\n"
           << "  \"end_time\": " << run.end_time << ",\n"
           << "  \"marker_count_at_start\": " << restart.markers.size()
           << ",\n  \"physics_task\": " << json_string(task_name(run.task))
           << ",\n  \"universe\": "
           << json_string(run.universe == UniverseMode::eds_expanding
                  ? "eds" : "static")
           << ",\n  \"poisson_source\": "
           << json_string(run.universe == UniverseMode::eds_expanding
                  ? "density_contrast" : "total_density") << ",\n"
           << "  \"poisson_reconstruction\": \"constant_line_mass\",\n"
           << "  \"collision_model\": "
           << json_string(collision_model_name(run))
           << ",\n"
           << "  \"collision_strength\": "
           << (material_collisions_enabled(run)
                  ? run.material_collision_strength : run.collision_strength)
           << ",\n  \"maximum_collision_interval\": "
           << run.maximum_collision_interval
           << ",\n  \"collision_timescale_fraction\": "
           << run.collision_timescale_fraction
           << ",\n  \"subhalo_adaptive_cell_timestep\": "
           << (run.subhalo_adaptive_cell_timestep ? "true" : "false")
           << ",\n  \"subhalo_adaptive_batch_substeps\": "
           << run.subhalo_adaptive_batch_substeps
           << ",\n  \"material_adaptive_shell_timestep\": "
           << (run.material_adaptive_shell_timestep ? "true" : "false")
           << ",\n  \"collision_lanes\": "
           << (material_collisions_enabled(run) ? 0 : run.collision_lanes)
           << ",\n  \"material_sigma0\": " << run.material_sigma0
           << ",\n  \"material_sigma_power\": " << run.material_sigma_power
           << ",\n  \"material_sigma_u0\": " << run.material_sigma_u0
           << ",\n  \"material_yukawa_alpha\": "
           << run.material_yukawa_alpha
           << ",\n  \"material_yukawa_mchi\": "
           << run.material_yukawa_mchi
           << ",\n  \"material_yukawa_mphi\": "
           << run.material_yukawa_mphi
           << ",\n  \"mpi_ranks\": "
#ifdef VENUS_GPU_MPI_RUNNER
           << world_ranks
#else
           << 1
#endif
           << (run.coupled_receiver_clocks
               ? ",\n  \"coupled_clock_policy\": \"gravitational_receiver_mass_rms_hidden_cpu_cadence\"" : "")
           << (run.collision_prestep_coordinate_due
                ? ",\n  \"collision_prestep_coordinate_due\": true" : "")
           << "\n}\n";
    output.flush();
    if (!output) {
        throw std::runtime_error("Failed writing GPU production metadata.");
    }
    output.close();
    fs::rename(temporary, directory / "metadata.json");
}

// Appending to a task directory requires the same physical task, universe,
// Poisson source convention, and collision model as its existing metadata.
void require_matching_existing_metadata(
    const fs::path& directory,
    const Options& run) {
    const fs::path path = directory / "metadata.json";
    if (run.task == ProductionTask::legacy_subhalo_auto) {
        return;
    }
    if (!fs::exists(path)) {
        if (fs::exists(directory) && !fs::is_empty(directory)) {
            throw std::runtime_error(
                "A task-specific GPU executable requires an empty output "
                "directory when metadata.json is absent.");
        }
        return;
    }
    std::ifstream input(path);
    std::ostringstream content;
    content << input.rdbuf();
    if (!input && !input.eof()) {
        throw std::runtime_error(
            "Could not read existing GPU production metadata.");
    }
    const std::string text = content.str();
    const std::string expected_task =
        std::string("\"physics_task\": ") + json_string(task_name(run.task));
    const std::string expected_universe =
        std::string("\"universe\": ")
        + json_string(run.universe == UniverseMode::eds_expanding
              ? "eds" : "static");
    const std::string expected_poisson_source =
        std::string("\"poisson_source\": ")
        + json_string(run.universe == UniverseMode::eds_expanding
              ? "density_contrast" : "total_density");
    const std::string expected_collision_model =
        std::string("\"collision_model\": ")
        + json_string(collision_model_name(run));
    if (text.find(expected_task) == std::string::npos
        || text.find(expected_universe) == std::string::npos
        || text.find(expected_poisson_source) == std::string::npos
        || text.find(expected_collision_model) == std::string::npos) {
        throw std::runtime_error(
            "Existing output metadata belongs to a different GPU physics task "
            "or uses a different universe, Poisson source, or collision model.");
    }
}

// A checkpoint may be continued in its existing output directory.  Every
// saved diagnostic must then precede that checkpoint; otherwise appending
// would join two different time intervals into one history.
void require_output_precedes_restart(
    const fs::path& directory, double restart_time) {
    const auto check_csv = [&](const fs::path& path) {
        if (!fs::exists(path)) return;
        std::ifstream input(path);
        std::string line;
        if (!input || !std::getline(input, line)) {
            throw std::runtime_error(
                "Could not read existing diagnostic: " + path.string());
        }
        std::stringstream header(line);
        std::string token;
        std::size_t time_column = 0;
        bool found_time = false;
        while (std::getline(header, token, ',')) {
            if (token == "time" || token == "time\r") {
                found_time = true;
                break;
            }
            ++time_column;
        }
        if (!found_time) {
            throw std::runtime_error(
                "Existing diagnostic has no time column: " + path.string());
        }
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            std::stringstream row(line);
            for (std::size_t column = 0; column <= time_column; ++column) {
                if (!std::getline(row, token, ',')) {
                    throw std::runtime_error(
                        "Missing diagnostic time in " + path.string());
                }
            }
            std::size_t consumed = 0;
            const double saved_time = std::stod(token, &consumed);
            if (!std::isfinite(saved_time)
                || token.find_first_not_of(" \t\r", consumed)
                    != std::string::npos) {
                throw std::runtime_error(
                    "Invalid diagnostic time in " + path.string());
            }
            const double tolerance = 128.0
                * std::numeric_limits<double>::epsilon()
                * std::max({1.0, std::abs(restart_time), std::abs(saved_time)});
            if (saved_time > restart_time + tolerance) {
                std::ostringstream error;
                error << std::setprecision(17) << "Existing diagnostic "
                      << path.string() << " contains time " << saved_time
                      << " later than restart time " << restart_time << '.';
                throw std::runtime_error(error.str());
            }
        }
        if (!input.eof()) {
            throw std::runtime_error(
                "Failed reading existing diagnostic: " + path.string());
        }
    };
    check_csv(directory / "history_gpu.csv");
    check_csv(directory / "collision_history_gpu.csv");
    for (const auto& entry : fs::directory_iterator(directory)) {
        const std::string name = entry.path().filename().string();
        if (entry.is_regular_file() && name.compare(0, 8, "profile_") == 0
            && entry.path().extension() == ".csv") {
            check_csv(entry.path());
        }
    }
}

// Gather accepted or current diagnostic particle state together with its
// epoch, initialization provenance, sampler, and carried event clocks.
// The caller's completion/safety status determines continuation eligibility.
venus_gpu::RestartState checkpoint_state(
    venus_gpu::WpicState& state,
    const venus_gpu::RestartState& provenance,
    double time,
    std::int64_t step,
    const venus_gpu::CollisionSamplerState& collision_sampler,
    double collision_age,
    double collision_time,
    double potential_time) {
    venus_gpu::RestartState restart;
    restart.markers = gather_checkpoint_markers(state);
    if (!root_rank()) {
        return restart;
    }
    restart.initialization_seed = provenance.initialization_seed;
    restart.initialization_counter = provenance.initialization_counter;
    restart.collision_sampler = collision_sampler;
    // Preserve the potential-event sampler clock for CPU/GPU checkpoint
    // interchange; this driver schedules its field updates independently.
    restart.potential_event_age = provenance.potential_event_age;
    restart.collision_event_age = collision_age;
    restart.collision_timescale_estimate = collision_time;
    restart.potential_timescale_estimate = potential_time;
    restart.time = time;
    restart.step = step;
    return restart;
}

// Publish a portable checkpoint on root after all MPI ranks participate in
// the particle gather. This operation preserves the source initialization keys.
void checkpoint(
    const fs::path& directory,
    venus_gpu::WpicState& state,
    const venus_gpu::RestartState& provenance,
    double time,
    std::int64_t step,
    const venus_gpu::CollisionSamplerState& collision_sampler,
    double collision_age,
    double collision_time,
    double potential_time) {
    const auto restart = checkpoint_state(state, provenance, time, step,
        collision_sampler, collision_age, collision_time, potential_time);
    if (root_rank()) {
        venus_gpu::write_restart(
            (directory / "restart_state.bin").string(), restart);
    }
}

// Existing selected particle/field groups must precede the supplied
// restart just as scalar/profile diagnostics do, preventing a mixed chronology.
void require_phase_space_precedes_restart(
    const fs::path& directory, double restart_time) {
    const fs::path snapshots = directory / "phase_space";
    if (!fs::exists(snapshots)) return;
    for (const auto& entry : fs::directory_iterator(snapshots)) {
        if (!entry.is_directory()
            || entry.path().filename().string().find(".tmp") != std::string::npos) {
            continue;
        }
        const fs::path path = entry.path() / "manifest.json";
        std::ifstream input(path);
        std::ostringstream content;
        content << input.rdbuf();
        const std::string text = content.str();
        const auto key = text.find("\"time\": ");
        if (!input || key == std::string::npos) {
            throw std::runtime_error("Invalid phase-space manifest: " + path.string());
        }
        const double saved_time = std::stod(text.substr(key + 8));
        const double tolerance = 128.0 * std::numeric_limits<double>::epsilon()
            * std::max({1.0, std::abs(saved_time), std::abs(restart_time)});
        if (!std::isfinite(saved_time) || saved_time > restart_time + tolerance) {
            throw std::runtime_error("Existing phase-space snapshot " + path.string()
                + " contains time later than restart time.");
        }
    }
}

// Publish particles, their actual field, and content hashes as one group.
// The manifest records both requested profile time and actual accepted time.
void write_phase_space_snapshot(
    const Options& run,
    std::int64_t index,
    const venus_gpu::RestartState& restart,
    const venus_gpu::FieldHost& field,
    const venus_gpu::RadialMesh& mesh,
    const venus_gpu::VlasovPoissonOptions& options,
    const std::string& source_restart_sha256,
    const std::string& executable_sha256) {
    std::ostringstream name;
    name << "snapshot_" << std::setw(5) << std::setfill('0') << index;
    const fs::path target = run.output / "phase_space" / name.str();
    const fs::path temporary = target.string() + ".tmp";
    // An interrupted write has no published directory. Rebuild its two files
    // and receipt, then install the complete group with one directory rename.
    if (fs::exists(temporary)) fs::remove_all(temporary);
    fs::create_directories(temporary);
    venus_gpu::write_restart((temporary / "particles.bin").string(), restart);
    venus_gpu::driver::write_phase_space_field_json(
        (temporary / "field.json").string(), restart.time, field, mesh, options,
        run.universe == UniverseMode::eds_expanding);
    const auto particle_hash = ev::integrity::sha256_file(temporary / "particles.bin");
    const auto field_hash = ev::integrity::sha256_file(temporary / "field.json");
    if (fs::exists(target)) {
        // Preserve an existing selected endpoint across exact continuations.
        // A different state at that time requires a distinct output directory.
        if (ev::integrity::sha256_file(target / "particles.bin") != particle_hash
            || ev::integrity::sha256_file(target / "field.json") != field_hash) {
            throw std::runtime_error("Existing phase-space snapshot differs: "
                + target.string());
        }
        fs::remove_all(temporary);
        return;
    }
    std::ofstream manifest(temporary / "manifest.json");
    manifest << std::setprecision(17)
        << "{\n  \"schema\": \"VENUS_phase_space_snapshot_1\",\n"
        << "  \"time\": " << restart.time
        << ",\n  \"requested_time\": " << index * run.output_interval
        << ",\n  \"step\": " << restart.step
        << ",\n  \"marker_count\": " << restart.markers.size()
        << ",\n  \"particles_file\": \"particles.bin\",\n"
        << "  \"particles_sha256\": " << json_string(particle_hash)
        << ",\n  \"field_file\": \"field.json\",\n"
        << "  \"field_sha256\": " << json_string(field_hash)
        << ",\n  \"source_restart\": " << json_string(run.restart.string())
        << ",\n  \"source_restart_sha256\": " << json_string(source_restart_sha256)
        << ",\n  \"executable_sha256\": " << json_string(executable_sha256)
        << ",\n  \"run_metadata_sha256\": "
        << json_string(ev::integrity::sha256_file(run.output / "metadata.json"))
        << ",\n  \"invocation\": [";
    for (std::size_t i = 0; i < run.invocation.size(); ++i) {
        if (i) manifest << ',';
        manifest << json_string(run.invocation[i]);
    }
    manifest << "]\n}\n";
    manifest.flush();
    if (!manifest) throw std::runtime_error("Failed writing phase-space receipt.");
    manifest.close();
    fs::rename(temporary, target);
}

}  // namespace

// Validate and load one physical task, initialize its device modules and
// carried clocks, then alternate characteristic steps with due local batches.
int venus_gpu::driver::run_production_task(
    int argc,
    char** argv,
    ProductionTask task) {
    try {
#ifdef VENUS_GPU_MPI_RUNNER
        MpiSession mpi_session(argc, argv);
#endif
        std::signal(SIGUSR1, request_stop);
        std::signal(SIGTERM, request_stop);
        const Options run = parse_options(argc, argv, task);
        if (run.show_help) {
            if (root_rank()) {
                print_usage(argv[0], task);
            }
            return 0;
        }
        require_matching_existing_metadata(run.output, run);
#ifdef VENUS_GPU_MPI_RUNNER
        venus_gpu::bind_cuda_device_to_mpi_local_rank(MPI_COMM_WORLD);
#else
        venus_gpu::require_cuda_device();
#endif
        if (root_rank()) {
            fs::create_directories(run.output);
        }
#ifdef VENUS_GPU_MPI_RUNNER
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        venus_gpu::RestartState source =
            venus_gpu::read_restart(run.restart.string());
        if (!(run.end_time > source.time)) {
            throw std::runtime_error("End time must follow restart time.");
        }
        require_output_precedes_restart(run.output, source.time);
        require_phase_space_precedes_restart(run.output, source.time);
        // The production characteristic path accepts active particles.
        // Subhalo encounters additionally require positive dispersion and the
        // same E_internal=3*m*sigma^2/2 relation used after capture.
        for (const venus_gpu::Marker& marker : source.markers) {
            if (marker.state != venus_gpu::MarkerState::active) {
                throw std::runtime_error(
                    "Current GPU KDK production requires active markers.");
            }
            if (subhalo_collisions_enabled(run)) {
                const double expected_internal_energy = 1.5 * marker.mass
                    * marker.internal_velocity_dispersion
                    * marker.internal_velocity_dispersion;
                const double internal_scale = std::max({
                    1.0e-300,
                    std::abs(expected_internal_energy),
                    std::abs(marker.internal_energy),
                });
                if (!(marker.internal_velocity_dispersion > 0.0)
                    || std::abs(
                           marker.internal_energy - expected_internal_energy)
                        > 2048.0 * std::numeric_limits<double>::epsilon()
                            * internal_scale) {
                    throw std::runtime_error(
                        "Collisional GPU production requires each active "
                        "marker to satisfy E_int=3*m*sigma_s^2/2 with "
                        "sigma_s>0.");
                }
            }
        }
        if (root_rank()) {
            // Keep the initial source metadata immutable across continuations.
            if (!fs::exists(run.output / "radial_mesh_contract.txt")) {
                write_mesh_contract(run.output, run);
            }
            if (!fs::exists(run.output / "metadata.json")) {
                write_metadata(run.output, run, source);
            }
        }

        const venus_gpu::RadialMesh mesh =
            venus_gpu::RadialMesh::from_edges(logarithmic_edges(run));
#ifdef VENUS_GPU_MPI_RUNNER
        // Distribute the input sequence by rank once. Later collision
        // migration changes ownership while IDs retain particle identity.
        std::vector<venus_gpu::Marker> local_markers;
        local_markers.reserve(
            (source.markers.size() + static_cast<std::size_t>(world_ranks) - 1)
            / static_cast<std::size_t>(world_ranks));
        for (std::size_t i = static_cast<std::size_t>(world_rank);
             i < source.markers.size();
             i += static_cast<std::size_t>(world_ranks)) {
            local_markers.push_back(source.markers[i]);
        }
        venus_gpu::WpicState state(local_markers);
#else
        venus_gpu::WpicState state(source.markers);
#endif
        // Free the second host copy before the long device-resident evolution.
        source.markers.clear();
        source.markers.shrink_to_fit();
#ifdef VENUS_GPU_MPI_RUNNER
        venus_gpu::MpiVlasovPoissonModule vp(mesh, MPI_COMM_WORLD);
        std::unique_ptr<venus_gpu::MpiSubhaloCollisionModule>
            subhalo_collisions;
        std::unique_ptr<venus_gpu::MpiCollisionModule> material_collisions;
        if (subhalo_collisions_enabled(run)) {
            subhalo_collisions.reset(
                new venus_gpu::MpiSubhaloCollisionModule(mesh, MPI_COMM_WORLD));
        }
        if (material_collisions_enabled(run)) {
            material_collisions.reset(
                new venus_gpu::MpiCollisionModule(mesh, MPI_COMM_WORLD));
        }
#else
        venus_gpu::VlasovPoissonModule vp(mesh);
        std::unique_ptr<venus_gpu::SubhaloCollisionModule> subhalo_collisions;
        std::unique_ptr<venus_gpu::CollisionModule> material_collisions;
        if (subhalo_collisions_enabled(run)) {
            subhalo_collisions.reset(new venus_gpu::SubhaloCollisionModule(mesh));
        }
        if (material_collisions_enabled(run)) {
            material_collisions.reset(new venus_gpu::CollisionModule(mesh));
        }
#endif
        const auto vp_control = vp_options(run);
        std::unique_ptr<venus_gpu_receiver::CoupledReceiverClocks> receiver_clocks;
        if (run.coupled_receiver_clocks) {
            receiver_clocks.reset(new venus_gpu_receiver::CoupledReceiverClocks(
                run.output.string(), run.coupled_field_trace));
        }
        const auto inner_face_potential = [&]() {
#ifdef VENUS_GPU_MPI_RUNNER
            throw std::runtime_error("Single-GPU receiver observation required.");
            return 0.;
#else
            return vp.download_inner_face_potential();
#endif
        };
        double field_inner_potential = 0.;
        const auto subhalo_collision_control = collision_options(run);
        const auto material_collision_control = material_collision_options(run);
        // Restore the actual sampler and accumulated age from the restart.
        // Resetting either would change the next stochastic batch or its time.
        venus_gpu::CollisionSamplerState collision_sampler =
            source.collision_sampler;
        double time = source.time;
        std::int64_t step = source.step;
        double collision_age = source.collision_event_age;
        double collision_time = source.collision_timescale_estimate;
        double potential_time = source.potential_timescale_estimate;
        // Observe a field pair with the configured receiver measure, or
        // retain the maximum-norm clock. Store bounded context if its closure
        // fails, preserving the original exception and exact sampler epoch.
        const auto selected_field_clock = [&](const venus_gpu::FieldHost& before,
                const venus_gpu::FieldHost& after, double interval,
                double previous_inner, double current_inner, const char* phase,
                std::int64_t observed_step, double observed_time) {
            const double maximum = minimum_potential_change_timescale(before,after,interval);
            if (!receiver_clocks) return maximum;
            try {
                return receiver_clocks->observe(state,before,after,mesh,vp_control,
                    previous_inner,current_inner,interval,maximum,phase,observed_step,observed_time);
            } catch (...) {
                try {
                    std::ofstream record(run.output / "receiver_failure_context.json");
                    record<<std::setprecision(17)
                        <<"{\"diagnostic_only\":true,\"phase\":"<<json_string(phase)
                        <<",\"state_stage\":"<<json_string(std::string(phase)=="trial"?"trial_drift_before_final_kick":"post_collision_update")
                        <<",\"time_argument\":"<<observed_time<<",\"step\":"<<observed_step
                        <<",\"interval\":"<<interval<<",\"collision_age\":"<<collision_age
                        <<",\"collision_seed\":"<<collision_sampler.seed
                        <<",\"collision_next_counter\":"<<collision_sampler.next_counter
                        <<",\"initialization_seed\":"<<source.initialization_seed
                        <<",\"initialization_counter\":"<<source.initialization_counter<<"}\n";
                    record.flush();
                } catch (...) {}
                throw;
            }
        };
        std::string snapshot_source_hash;
        std::string snapshot_executable_hash;
        if (root_rank() && !run.phase_space_snapshot_times.empty()) {
            snapshot_source_hash = ev::integrity::sha256_file(run.restart);
            // Linux supplies the running executable even when argv[0] came
            // from PATH. The explicit path supports other host test systems.
            const fs::path executable = fs::exists("/proc/self/exe")
                ? fs::path("/proc/self/exe") : fs::path(argv[0]);
            snapshot_executable_hash = ev::integrity::sha256_file(executable);
        }

        vp.deposit_and_solve(state, time, vp_control);
#ifdef VENUS_GPU_MPI_RUNNER
        venus_gpu::FieldHost field = vp.download_global_field();
#else
        venus_gpu::FieldHost field = vp.download_field();
#endif
        if (receiver_clocks) field_inner_potential = inner_face_potential();
        // Initialize an unavailable collision-time estimate through the
        // operator's minimum-positive-interval diagnostic call, using a copy
        // of the sampler so its diagnostic epoch is not committed.
        if (subhalo_collisions_enabled(run)
            && !std::isfinite(collision_time)) {
            venus_gpu::CollisionSamplerState diagnostic_sampler =
                collision_sampler;
            const auto diagnostic = subhalo_collisions->apply(
                state,
                std::numeric_limits<double>::min(),
                subhalo_collision_control,
                diagnostic_sampler);
            collision_time = diagnostic.minimum_collision_timescale;
        }
        const auto save_selected_phase_space = [&]() {
            const auto index = venus_gpu::driver::phase_space_snapshot_index(
                run.phase_space_snapshot_times, time, run.output_interval);
            if (index < 0) return;
            // Every rank participates in the ordered marker gather. Root
            // serializes the already reconstructed endpoint field; no solve,
            // collision sampling, or additional orbit update occurs here.
            const auto saved = checkpoint_state(state, source, time, step,
                collision_sampler, collision_age, collision_time, potential_time);
            if (root_rank()) write_phase_space_snapshot(run, index, saved,
                field, mesh, vp_control, snapshot_source_hash,
                snapshot_executable_hash);
        };

        // Continue existing histories by append after their chronology has
        // been checked. Separate schemas record subhalo and material budgets.
        const bool new_history = root_rank()
            && !fs::exists(run.output / "history_gpu.csv");
        std::ofstream history;
        if (root_rank()) {
            history.open(
                run.output / "history_gpu.csv",
                new_history ? std::ios::trunc : std::ios::app);
            if (!history) {
                throw std::runtime_error("Could not open GPU history output.");
            }
        }
        if (root_rank() && new_history) {
            history << "step,time,dt,mobile_mass,marker_count,timestep_limit,"
                       "collision_age,collision_time,potential_time,wall_seconds\n";
        }
        const bool emit_collision_history = collisions_enabled(run)
            || run.task == ProductionTask::legacy_subhalo_auto;
        const bool new_collision = root_rank() && emit_collision_history &&
            !fs::exists(run.output / "collision_history_gpu.csv");
        std::ofstream collision_history;
        if (root_rank() && emit_collision_history) {
            collision_history.open(
                run.output / "collision_history_gpu.csv",
                new_collision ? std::ios::trunc : std::ios::app);
            if (!collision_history) {
                throw std::runtime_error(
                    "Could not open GPU collision-history output.");
            }
        }
        if (root_rank() && new_collision && subhalo_collisions_enabled(run)) {
            collision_history
                << "step,time,dt,substeps,trial_pairs,"
                   "skipped_already_collided_trials,accepted_pairs,"
                   "scattering_events,capture_events,removed_markers,"
                   "represented_scattering_mass,represented_capture_mass,"
                   "orbital_kinetic_energy_change,internal_energy_change,"
                   "max_pair_momentum_error,max_pair_total_energy_error,"
                   "min_collision_time,max_dr_over_mfp,marker_count\n";
        } else if (root_rank() && new_collision
                   && material_collisions_enabled(run)) {
            collision_history
                << "step,time,dt,substeps,shell_visits,trial_pairs,"
                   "accepted_pairs,active_markers,shells,nonlocal_shells,"
                   "radial_momentum_error,kinetic_energy_error,"
                   "max_pair_momentum_error,max_pair_kinetic_energy_error,"
                   "max_shell_width_over_mfp,marker_count\n";
        } else if (root_rank() && new_collision) {
            // Preserve the compatibility file contract for a
            // legacy strength-zero run, although no rows will be written.
            collision_history
                << "step,time,dt,substeps,trial_pairs,accepted_pairs\n";
        }

        // Map the restart epoch to the existing profile grid. A checkpoint
        // between callbacks retains the earlier completed profile at that index.
        std::int64_t output_index = static_cast<std::int64_t>(
            std::floor((time + 1.0e-11 * std::max(1.0, std::abs(time)))
                / run.output_interval));
        if (root_rank()) {
            std::ostringstream name;
            name << "profile_" << std::setw(5) << std::setfill('0')
                 << output_index << ".csv";
            // A walltime checkpoint can lie between output times. Preserve
            // the completed earlier profile at that index when continuing.
            if (!fs::exists(run.output / name.str())) {
                write_profile(run.output, output_index, time, field, mesh);
            }
        }
        double next_output = (output_index + 1) * run.output_interval;
        save_selected_phase_space();
        const auto wall_start = std::chrono::steady_clock::now();
        venus_gpu::WpicOrbitRollbackBuffer rollback;
        bool locality_stop = false;
        bool cadence_stop = false;

        // A batch consumes the full carried age at fixed coordinate time.
        // Check its deadline before sampling, apply the physical operator,
        // then rebuild gravity and the field-change time from its new state.
        auto apply_collision = [&](double deadline) {
            if (!deadline_safe(collision_age, deadline)) {
                cadence_stop = true;
                return;
            }
            const venus_gpu::FieldHost before = field;
            const double previous_inner_potential = field_inner_potential;
            const double applied = collision_age;
            double maximum_width_over_mfp = 0.0;
            if (subhalo_collisions_enabled(run)) {
                const auto diagnostic = subhalo_collisions->apply(
                    state,
                    applied,
                    subhalo_collision_control,
                    collision_sampler);
                collision_time = diagnostic.minimum_collision_timescale;
                maximum_width_over_mfp =
                    diagnostic.maximum_cell_width_over_mean_free_path;
                if (root_rank()) {
                    collision_history << std::setprecision(17)
                    << step << ',' << time << ',' << applied << ','
                    << diagnostic.substeps << ',' << diagnostic.trial_pairs
                    << ',' << diagnostic.skipped_already_collided_trials << ','
                    << diagnostic.accepted_pairs << ','
                    << diagnostic.scattering_events << ','
                    << diagnostic.capture_events << ','
                    << diagnostic.removed_markers << ','
                    << diagnostic.represented_scattering_mass << ','
                    << diagnostic.represented_capture_mass << ','
                    << diagnostic.orbital_kinetic_energy_change << ','
                    << diagnostic.internal_energy_change << ','
                    << diagnostic.maximum_pair_momentum_error << ','
                    << diagnostic.maximum_pair_total_energy_error << ','
                    << diagnostic.minimum_collision_timescale << ','
                    << diagnostic.maximum_cell_width_over_mean_free_path << ','
                    << diagnostic.active_markers_after << '\n';
                }
            } else if (material_collisions_enabled(run)) {
                const auto diagnostic = material_collisions->apply(
                    state,
                    applied,
                    material_collision_control,
                    collision_sampler);
                // The fixed-occupancy material operator has no unbiased
                // instantaneous collision-time estimator.  The task contract
                // therefore keeps a fixed event interval and records infinity
                // in the portable restart field.
                collision_time = std::numeric_limits<double>::infinity();
                maximum_width_over_mfp =
                    diagnostic.maximum_shell_width_over_mean_free_path;
                if (root_rank()) {
                    collision_history << std::setprecision(17)
                    << step << ',' << time << ',' << applied << ','
                    << diagnostic.substeps << ',' << diagnostic.shell_visits
                    << ',' << diagnostic.trial_pairs << ','
                    << diagnostic.accepted_pairs << ','
                    << diagnostic.active_markers << ',' << diagnostic.shells
                    << ',' << diagnostic.nonlocal_shells << ','
                    << diagnostic.radial_momentum_error << ','
                    << diagnostic.kinetic_energy_error << ','
                    << diagnostic.max_pair_momentum_error << ','
                    << diagnostic.max_pair_kinetic_energy_error << ','
                    << diagnostic.maximum_shell_width_over_mean_free_path << ','
                    << diagnostic.active_markers << '\n';
                }
            } else {
                throw std::runtime_error(
                    "Internal error: collision deadline in a collisionless task.");
            }
            // The applied interval has now been consumed. Capture or velocity
            // changes require a new field/moment observation before another step.
            collision_age = 0.0;
            vp.deposit_and_solve(state, time, vp_control);
#ifdef VENUS_GPU_MPI_RUNNER
            field = vp.download_global_field();
#else
            field = vp.download_field();
#endif
            if (receiver_clocks) field_inner_potential = inner_face_potential();
            potential_time = selected_field_clock(before,field,applied,
                previous_inner_potential,field_inner_potential,"collision",step,time);
            if (root_rank()) {
                collision_history.flush();
                if (!collision_history) {
                    throw std::runtime_error(
                        "Failed writing GPU collision history.");
                }
            }
            // Locality is evaluated by the operator's incoming physical-cell
            // diagnostic. A stop is distinct from a pair-probability/CFL error;
            // the current state may already include the completed batch.
            if (root_rank() && maximum_width_over_mfp > 1.0) {
                std::cerr << "LOCALITY WARNING t=" << time
                          << " max_dr_over_mfp="
                          << maximum_width_over_mfp
                          << '\n';
            }
            if (subhalo_collisions_enabled(run)
                && run.locality_maximum_ratio > 0.0
                && maximum_width_over_mfp > run.locality_maximum_ratio) {
                locality_stop = true;
            }
        };

        double previous_collision_decision_time = time;
        while (time < run.end_time && !globally_stop_requested()
               && !locality_stop && !cadence_stop) {
            // Recompute orbit, dynamical, and collision bounds from the
            // current field before choosing the next characteristic interval.
            const double timestep_limit = choose_timestep(field, mesh, time, run);
            const double dynamical_time =
                minimum_dynamical_timescale(field, mesh);
            double deadline = selected_collision_interval(
                run, collision_time, dynamical_time, potential_time);
            if (collisions_enabled(run)) {
                // A positive deadline must be resolvable at the current
                // coordinate time before accumulating or sampling an interval.
                const double coordinate_resolution =
                    128.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, std::abs(time));
                if (!(deadline > coordinate_resolution)
                    || !std::isfinite(deadline)) {
                    if (root_rank()) {
                        std::cerr
                            << "CADENCE STOP: selected collision interval="
                            << std::setprecision(17) << deadline
                            << " is below time-coordinate resolution="
                            << coordinate_resolution << " at t=" << time
                            << "; checkpointing without sampling a batch.\n";
                    }
                    cadence_stop = true;
                    break;
                }
            }
            // The optional pre-step decision consumes a safely due positive
            // carried age at the present state. Other overdue/unresolved
            // classifications stop before an additional characteristic trial.
            if (collisions_enabled(run) && run.collision_prestep_coordinate_due) {
                const auto due = ev::assess_weighted_pic_pre_step_collision_cadence(
                    collision_age, deadline, time, previous_collision_decision_time);
                previous_collision_decision_time = time;
                if (due.status != ev::WeightedPICPreStepCollisionStatus::advance) {
                    if (root_rank()) {
                        std::cerr << "CADENCE PRE-STEP: t=" << std::setprecision(17)
                            << time << " age=" << collision_age
                            << " deadline=" << deadline
                            << " remaining=" << due.remaining_interval
                            << " coordinate_resolution=" << due.coordinate_resolution
                            << " action=" << ev::weighted_pic_pre_step_collision_status_name(due.status)
                            << '\n';
                    }
                    if (due.status != ev::WeightedPICPreStepCollisionStatus::apply_collision_now) {
                        cadence_stop = true;
                        break;
                    }
                    apply_collision(deadline);
                    continue;
                }
            }
            // Land on the earliest orbit cap, requested end, next profile,
            // or remaining collision deadline. These clocks share physical time.
            double dt = std::min({timestep_limit, run.end_time - time,
                std::max(0.0, next_output - time)});
            if (collisions_enabled(run)) {
                dt = std::min(dt, std::max(0.0, deadline - collision_age));
            }
            if (!(dt > 0.0)) {
                if (collisions_enabled(run)
                    && collision_age >= deadline
                        - 1.0e-12 * std::max(1.0, deadline)) {
                    apply_collision(deadline);
                    continue;
                }
                next_output += run.output_interval;
                continue;
            }

            // A changing field can shorten a subhalo deadline during KDK.
            // Save the incoming orbit so an unsafe trial can be rejected
            // without committing particle motion or consuming random draws.
            const bool transactional = subhalo_collisions_enabled(run)
                && run.collision_timescale_fraction > 0.0;
            if (transactional) {
                rollback.capture(state);
            }
            bool apply_before = false;
            bool apply_after_retried = false;
            double endpoint_inner_potential = field_inner_potential;
            int retry_depth = 0;
            venus_gpu::FieldHost endpoint;
            // Probe the drifted endpoint field before the final kick. Its
            // measured variation and enclosed mass can tighten the deadline.
            while (true) {
                vp.begin_kdk_with_current_field(state, time, dt, vp_control);
                vp.deposit_and_solve(state, time + dt, vp_control);
#ifdef VENUS_GPU_MPI_RUNNER
                endpoint = vp.download_global_field();
#else
                endpoint = vp.download_field();
#endif
                if (receiver_clocks) endpoint_inner_potential = inner_face_potential();
                const double endpoint_potential = std::min(
                    potential_time, selected_field_clock(field,endpoint,dt,
                        field_inner_potential,endpoint_inner_potential,"trial",step,time));
                const double endpoint_dynamical =
                    minimum_dynamical_timescale(endpoint, mesh);
                const double endpoint_deadline = selected_collision_interval(
                    run, collision_time, endpoint_dynamical,
                    endpoint_potential);
                if (transactional && run.coupled_receiver_clocks) {
                    const auto cadence = ev::assess_weighted_pic_endpoint_collision_cadence(
                        collision_age,dt,time,run.maximum_collision_interval,
                        run.collision_timescale_fraction,
                        run.subhalo_adaptive_cell_timestep
                            ? std::numeric_limits<double>::infinity() : collision_time,
                        endpoint_dynamical,endpoint_potential);
                    receiver_clocks->decision(step,time,dt,collision_age,cadence);
                    // Restore both the incoming orbit and its stored field
                    // before applying a due batch or attempting a shorter step.
                    if (!cadence.accept_step) {
                        rollback.restore(state);
                        vp.solve_from_global_moments(field,time,vp_control);
                        ++retry_depth;
                        if (cadence.apply_collision_now) {
                            apply_before = true;
                            deadline = cadence.selected_interval;
                            break;
                        }
                        if (cadence.unresolved_overdue || retry_depth >= 64) {
                            cadence_stop = true;
                            break;
                        }
                        dt = cadence.retry_interval;
                        continue;
                    }
                    if (retry_depth > 0) apply_after_retried = true;
                }
                // The ordinary adaptive policy uses the same rollback idea,
                // with its own deadline safety and bounded shortening rule.
                if (transactional && !run.coupled_receiver_clocks
                    && !deadline_safe(collision_age + dt, endpoint_deadline)) {
                    rollback.restore(state);
                    vp.solve_from_global_moments(field, time, vp_control);
                    ++retry_depth;
                    const double coordinate_resolution =
                        128.0 * std::numeric_limits<double>::epsilon()
                        * std::max(1.0, std::abs(time));
                    const double left = endpoint_deadline - collision_age;
                    if (left <= coordinate_resolution
                        && deadline_safe(collision_age, endpoint_deadline)) {
                        apply_before = true;
                        deadline = endpoint_deadline;
                        break;
                    }
                    if (retry_depth >= 64 || dt <= coordinate_resolution) {
                        cadence_stop = true;
                        break;
                    }
                    dt = left > coordinate_resolution
                        ? std::min(0.875 * dt, 0.95 * left) : 0.5 * dt;
                    if (!(dt > 0.0) || !std::isfinite(dt)) {
                        cadence_stop = true;
                        break;
                    }
                    continue;
                }
                // Commit the accepted KDK endpoint before sampling encounters.
                // The opt-in policy applies after a shortened accepted retry,
                // as in the CPU cadence helper. The default policy retains
                // its post-step age/deadline test below.
                vp.finish_kdk_with_current_field(
                    state, time + dt, dt, vp_control);
                potential_time = endpoint_potential;
                break;
            }
            if (apply_before) {
                apply_collision(deadline);
                continue;
            }
            if (cadence_stop) {
                break;
            }
            // Advance coordinate time, step, and collision age only after
            // the endpoint has passed cadence checks and its kick is committed.
            time += dt;
            if (time >= run.end_time
                    - 1.0e-12 * std::max(1.0, run.end_time)
                && time <= run.end_time
                    + 1.0e-12 * std::max(1.0, run.end_time)) {
                time = run.end_time;
            }
            ++step;
            if (collisions_enabled(run)
                || run.task == ProductionTask::legacy_subhalo_auto) {
                collision_age += dt;
            }
            field = std::move(endpoint);
            if (receiver_clocks) field_inner_potential = endpoint_inner_potential;
            const bool final = time >= run.end_time
                - 1.0e-12 * std::max(1.0, run.end_time);
            deadline = selected_collision_interval(
                run,
                collision_time,
                minimum_dynamical_timescale(field, mesh),
                potential_time);
            // Apply encounters after a due accepted endpoint or final
            // landing. A shortened accepted coupled retry also consumes its
            // carried age according to the selected cadence policy.
            if (collisions_enabled(run)
                && (collision_age >= deadline
                        - 1.0e-12 * std::max(1.0, deadline)
                    || final || (run.coupled_receiver_clocks && apply_after_retried))) {
                apply_collision(deadline);
            }

            const bool output_due = time >= next_output
                - 1.0e-11 * std::max(1.0, next_output);
            const double wall = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - wall_start).count();
            // Publish the accepted orbit epoch and current operator ages
            // on the requested step/output/final callbacks.
            if (step % run.history_every == 0 || output_due || final) {
                const std::size_t markers = global_marker_count(state.size());
                if (root_rank()) {
                    history << std::setprecision(17)
                    << step << ',' << time << ',' << dt << ','
                    << field.diagnostics.deposited_mesh_mass
                        + field.diagnostics.hidden_inner_mass
                    << ',' << markers << ',' << timestep_limit << ','
                    << collision_age << ',' << collision_time << ','
                    << potential_time << ',' << wall << '\n';
                    history.flush();
                    if (!history) {
                        throw std::runtime_error("Failed writing GPU history.");
                    }
                    std::cout << std::setprecision(10)
                          << "VENUS GPU step=" << step << " t=" << time
                          << " dt=" << dt << " markers=" << markers
                          << " wall=" << wall << '\n';
                }
            }
            if (output_due || final) {
                // The force solve precedes the last KDK kick. Density and
                // tangential moments are unchanged by that radial kick, but
                // the saved radial mean, variance and energy flux must use
                // the completed endpoint velocities, as the restart does.
                vp.deposit_and_solve(state, time, vp_control);
#ifdef VENUS_GPU_MPI_RUNNER
                field = vp.download_global_field();
#else
                field = vp.download_field();
#endif
                if (receiver_clocks) field_inner_potential = inner_face_potential();
                ++output_index;
                if (root_rank()) {
                    write_profile(run.output, output_index, time, field, mesh);
                }
                save_selected_phase_space();
                next_output += run.output_interval;
            }
            // Checkpoint cadence is independent of profile cadence; save
            // the complete current particle/clock state at its accepted epoch.
            if (step % run.restart_every == 0 || final) {
                checkpoint(run.output, state, source, time, step,
                    collision_sampler, collision_age, collision_time,
                    potential_time);
            }
        }

        // Save the current state before reporting why evolution ended.
        // Safety-stop checkpoints preserve diagnosis; their 97/98 status
        // carries a different scientific meaning from completion or walltime.
        checkpoint(run.output, state, source, time, step, collision_sampler,
            collision_age, collision_time, potential_time);
        if (receiver_clocks) receiver_clocks->finish();
        if (globally_stop_requested()) {
            if (root_rank()) {
                std::ofstream(run.output / "walltime_checkpoint.ok")
                    << time << '\n';
            }
            return 99;
        }
        if (locality_stop) {
            if (root_rank()) {
                std::ofstream(run.output / "locality_stopped.ok")
                    << time << '\n';
            }
            return 98;
        }
        if (cadence_stop) {
            if (root_rank()) {
                std::ofstream(run.output / "cadence_stopped.ok")
                    << time << '\n';
            }
            return 97;
        }
        if (root_rank()) {
            std::ofstream(run.output / "completed.ok")
                << std::setprecision(17) << time << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VENUS GPU production error: " << error.what() << '\n';
        return 2;
    }
}
