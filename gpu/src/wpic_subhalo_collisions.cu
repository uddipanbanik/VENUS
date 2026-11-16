// Sample scattering and capture between unequal-mass Plummer subhalos.
// Inputs are device particle masses, orbits, internal dispersions/energies,
// a common radial mesh, elapsed collision time, and sampler seed/counter.
// The calibrated CPU table supplies total cross sections. Cell-local pair
// proposals sample those rates; scattering changes velocities, while capture
// combines mass and stores lost relative orbital energy as internal energy.
// Outputs are updated particles, advanced sampler state, and conservation,
// event, clock, and locality diagnostics. Each cell has one sequential owner
// within the GPU launch, preserving its prefix-sum and random-draw ordering.
// MPI supplies the global cell offset and any common-rate reduction.

#include "venus_gpu/subhalo_collisions.hpp"

#include "cuda_utils.cuh"
#include "venus_gpu_generated_subhalo_table.hpp"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace venus_gpu {
namespace {

constexpr int subhalo_threads = 256;

struct DeviceSubhaloOptions {
    double area_per_mass;
    double low_speed_regularization;
    double strength;
    double collision_cfl;
    int inner_cells;
    int lanes;
    std::uint64_t minimum_population;
    std::uint64_t resolved_cell_offset;
    int enable_scattering;
    int enable_capture;
};

// Dimensionless capture/scatter tables depend on mass ratio and speed
// scaled by the pair's internal dispersion. Their global bounds control
// proposal frequency separately from the interpolated physical rate.
struct DeviceSubhaloTable {
    const double* mass_ratio;
    const double* speed;
    const double* capture;
    const double* scatter;
    const double* scatter_high;
    double capture_bound;
    double scatter_bound;
};

// Collision-cell state refers to a sorted-index interval, its assigned
// volume/width, the sum of all pair majorants, and the fastest marker rate.
struct DeviceCell {
    std::uint64_t begin;
    std::uint64_t end;
    double volume;
    double width;
    double pair_majorant;
    double maximum_trial_frequency;
};

// Sort by physical cell/lane, then increasing internal dispersion and ID.
// This order makes the larger-index partner supply the pair's sigma maximum.
struct ProposalKey {
    std::uint64_t cell;
    double internal_velocity_dispersion;
    std::uint64_t id;
};

struct ProposalKeyLess {
    __host__ __device__ bool operator()(
        const ProposalKey& first,
        const ProposalKey& second) const {
        if (first.cell != second.cell) {
            return first.cell < second.cell;
        }
        if (first.internal_velocity_dispersion
            != second.internal_velocity_dispersion) {
            return first.internal_velocity_dispersion
                < second.internal_velocity_dispersion;
        }
        return first.id < second.id;
    }
};

// These sampled physical-cell rates estimate locality and collision time
// before lane splitting. They are separate from the rigorous proposal bound.
struct DevicePhysicalDiagnostic {
    double mean_free_path;
    double collision_timescale;
    double width_over_mean_free_path;
    double minimum_pair_velocity_scale;
    double maximum_pair_velocity_scale;
    std::uint64_t pair_samples;
    int nonlocal;
};

// Extensive event/energy counters add across cells. Worst-pair residuals
// and maximum local work remain extrema, retaining individual failures.
struct DeviceSubhaloStats {
    unsigned long long cell_visits;
    unsigned long long trial_pairs;
    unsigned long long skipped_already_collided_trials;
    unsigned long long accepted_pairs;
    unsigned long long scattering_events;
    unsigned long long capture_events;
    unsigned long long inner_scattering_events;
    unsigned long long inner_capture_events;
    unsigned long long removed_markers;
    double represented_scattering_mass;
    double represented_capture_mass;
    double orbital_energy_change;
    double internal_energy_change;
    unsigned long long max_pair_momentum_bits;
    unsigned long long max_pair_energy_bits;
    unsigned long long unfinished_cells;
    unsigned long long maximum_cell_substeps;
    int error_flag;
};

struct ActivePredicate {
    const std::uint8_t* state;
    __host__ __device__ bool operator()(std::uint64_t index) const {
        return state[index] == static_cast<std::uint8_t>(MarkerState::active);
    }
};

__host__ __device__ inline std::uint64_t mix64(std::uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

// Bit-for-bit copy of the current CPU subhalo counter stream.  A collision
// outcome is keyed by (epoch, cell, substep, trial, dimension), so scheduling
// CUDA blocks in a different order cannot change its random variates.
__host__ __device__ inline double subhalo_uniform(
    std::uint64_t seed,
    std::uint64_t epoch,
    std::uint64_t cell,
    std::uint64_t substep,
    std::uint64_t trial,
    std::uint64_t dimension) {
    const std::uint64_t counter =
        mix64(epoch)
        ^ mix64(cell + UINT64_C(0x100000001b3))
        ^ mix64(substep + UINT64_C(0x9e3779b9))
        ^ mix64(trial + UINT64_C(0x85ebca6b));
    const std::uint64_t word = mix64(
        seed ^ mix64(counter)
        ^ mix64(dimension + UINT64_C(0xd1b54a32d192ed03)));
    const double value = (static_cast<double>(word >> 11) + 0.5)
        * (1.0 / 9007199254740992.0);
    return fmin(value, 0.99999999999999989);
}

struct OriginPairVelocity {
    double x, y, z;
};

// At r=0 rotational invariance makes the velocity direction isotropic.  The
// reduced marker stores its speed; this vector is shared by rate and outcome.
__host__ __device__ inline OriginPairVelocity sample_origin_velocity(
    double radius, double radial_speed, double eta, double azimuth,
    std::uint64_t seed, std::uint64_t epoch, std::uint64_t cell,
    std::uint64_t substep, std::uint64_t trial, std::uint64_t dimension) {
    if (radius > 0.0) {
        const double vt = sqrt(2.0 * eta) / fmax(radius, 1.0e-300);
        return {radial_speed, vt * cos(azimuth), vt * sin(azimuth)};
    }
    const double mu = 2.0 * subhalo_uniform(
        seed, epoch, cell, substep, trial, dimension) - 1.0;
    const double phi = 2.0 * pi * subhalo_uniform(
        seed, epoch, cell, substep, trial, dimension + 1);
    const double transverse = sqrt(fmax(0.0, 1.0 - mu * mu));
    const double speed = fabs(radial_speed);
    return {speed * mu, speed * transverse * cos(phi),
            speed * transverse * sin(phi)};
}

__host__ __device__ inline double origin_pair_speed(
    OriginPairVelocity first, OriginPairVelocity second) {
    const double x = first.x - second.x;
    const double y = first.y - second.y;
    const double z = first.z - second.z;
    return sqrt(x * x + y * y + z * z);
}

__host__ __device__ inline int upper_index(
    const double* values,
    int count,
    double value) {
    int first = 0;
    int last = count;
    while (first < last) {
        const int middle = first + (last - first) / 2;
        if (value < values[middle]) {
            last = middle;
        } else {
            first = middle + 1;
        }
    }
    return first;
}

// Interpolate positive table values geometrically in logarithmic space.
// A zero endpoint uses a nonnegative linear interpolation to retain support.
__host__ __device__ inline double interpolate_positive_or_linear(
    double first,
    double second,
    double fraction) {
    if (first > 0.0 && second > 0.0) {
        return exp((1.0 - fraction) * log(first) + fraction * log(second));
    }
    return fmax(0.0, (1.0 - fraction) * first + fraction * second);
}

struct RatioBracket {
    int low;
    int high;
    double fraction;
};

// Bracket min(m1,m2)/max(m1,m2) on the logarithmic mass-ratio grid,
// using the nearest tabulated row at either end of its calibrated range.
__host__ __device__ inline RatioBracket ratio_bracket(
    double value,
    DeviceSubhaloTable table) {
    constexpr int count = generated_subhalo_table::mass_ratio_count;
    if (value <= table.mass_ratio[0]) {
        return {0, 0, 0.0};
    }
    if (value >= table.mass_ratio[count - 1]) {
        return {count - 1, count - 1, 0.0};
    }
    const int high = upper_index(table.mass_ratio, count, value);
    const int low = high - 1;
    const double fraction =
        (log(value) - log(table.mass_ratio[low]))
        / (log(table.mass_ratio[high]) - log(table.mass_ratio[low]));
    return {low, high, fraction};
}

// Interpolate along one logarithmic dimensionless-speed row. The caller
// supplies the separate high-speed continuation for each physical channel.
__host__ __device__ inline double interpolate_speed_row(
    double value,
    const double* row,
    DeviceSubhaloTable table) {
    constexpr int count = generated_subhalo_table::speed_count;
    if (value <= table.speed[0]) {
        return row[0];
    }
    if (value >= table.speed[count - 1]) {
        return row[count - 1];
    }
    const int high = upper_index(table.speed, count, value);
    const int low = high - 1;
    const double fraction =
        (log(value) - log(table.speed[low]))
        / (log(table.speed[high]) - log(table.speed[low]));
    return interpolate_positive_or_linear(row[low], row[high], fraction);
}

// Regulate the low-speed argument at the configured table floor. The
// calibrated capture channel has zero support beyond its final speed knot.
__host__ __device__ inline double capture_shape(
    double dimensionless_speed,
    double mass_ratio,
    DeviceSubhaloOptions options,
    DeviceSubhaloTable table) {
    const double regulated = fmax(
        dimensionless_speed, options.low_speed_regularization);
    constexpr int speed_count = generated_subhalo_table::speed_count;
    if (regulated >= table.speed[speed_count - 1]) {
        return 0.0;
    }
    const RatioBracket bracket = ratio_bracket(mass_ratio, table);
    const double low = interpolate_speed_row(
        regulated, table.capture + bracket.low * speed_count, table);
    if (bracket.low == bracket.high) {
        return low;
    }
    const double high = interpolate_speed_row(
        regulated, table.capture + bracket.high * speed_count, table);
    return interpolate_positive_or_linear(low, high, bracket.fraction);
}

// Continue the scattering table with its calibrated inverse-fourth-power
// high-speed tail, interpolating that tail's coefficient in mass ratio.
__host__ __device__ inline double scatter_shape(
    double dimensionless_speed,
    double mass_ratio,
    DeviceSubhaloOptions options,
    DeviceSubhaloTable table) {
    const double regulated = fmax(
        dimensionless_speed, options.low_speed_regularization);
    constexpr int speed_count = generated_subhalo_table::speed_count;
    const RatioBracket bracket = ratio_bracket(mass_ratio, table);
    if (regulated >= table.speed[speed_count - 1]) {
        const double coefficient = bracket.low == bracket.high
            ? table.scatter_high[bracket.low]
            : interpolate_positive_or_linear(
                table.scatter_high[bracket.low],
                table.scatter_high[bracket.high],
                bracket.fraction);
        const double squared = regulated * regulated;
        return coefficient / (squared * squared);
    }
    const double low = interpolate_speed_row(
        regulated, table.scatter + bracket.low * speed_count, table);
    if (bracket.low == bracket.high) {
        return low;
    }
    const double high = interpolate_speed_row(
        regulated, table.scatter + bracket.high * speed_count, table);
    return interpolate_positive_or_linear(low, high, bracket.fraction);
}

// The implemented pair speed scale is the square-root-mass-weighted RMS
// of the two internal dispersions. Use the actual evolving pair states.
__host__ __device__ inline double pair_velocity_scale(
    double first_mass,
    double second_mass,
    double first_sigma,
    double second_sigma) {
    const double first_weight = sqrt(first_mass);
    const double second_weight = sqrt(second_mass);
    return sqrt(
        (first_weight * first_sigma * first_sigma
         + second_weight * second_sigma * second_sigma)
        / (first_weight + second_weight));
}

// The sum of square roots of mass gives the geometric pair-area scaling
// selected by this unequal-Plummer calibration.
__host__ __device__ inline double pair_area(
    double first_mass,
    double second_mass,
    DeviceSubhaloOptions options) {
    const double root_sum = sqrt(first_mass) + sqrt(second_mass);
    return 0.25 * options.area_per_mass * root_sum * root_sum;
}

// Convert physical relative speed to the table's dimensionless argument,
// return enabled channel areas, and retain their speed-times-area majorant.
__host__ __device__ inline SubhaloCrossSections evaluate_pair(
    double relative_speed,
    double first_mass,
    double second_mass,
    double first_sigma,
    double second_sigma,
    DeviceSubhaloOptions options,
    DeviceSubhaloTable table) {
    SubhaloCrossSections result;
    result.pair_velocity_scale = pair_velocity_scale(
        first_mass, second_mass, first_sigma, second_sigma);
    result.pair_area = pair_area(first_mass, second_mass, options);
    const double ratio = fmin(first_mass, second_mass)
        / fmax(first_mass, second_mass);
    const double x = relative_speed / result.pair_velocity_scale;
    result.scattering = options.enable_scattering != 0
        ? options.strength * result.pair_area
            * scatter_shape(x, ratio, options, table)
        : 0.0;
    result.capture = options.enable_capture != 0
        ? options.strength * result.pair_area
            * capture_shape(x, ratio, options, table)
        : 0.0;
    // The table bounds x*shape(x). Multiplication by a pair speed scale
    // no larger than max(sigma1,sigma2) bounds the physical u*Sigma kernel.
    const double bound =
        (options.enable_scattering != 0 ? table.scatter_bound : 0.0)
        + (options.enable_capture != 0 ? table.capture_bound : 0.0);
    result.u_sigma_majorant = result.pair_area
        * fmax(first_sigma, second_sigma) * options.strength * bound
        * (1.0 + 128.0 * 2.2204460492503131e-16);
    return result;
}

__device__ inline void atomic_max_bits(
    unsigned long long* address,
    double value) {
    const unsigned long long desired = __double_as_longlong(value);
    unsigned long long old = *address;
    while (__longlong_as_double(static_cast<long long>(old)) < value) {
        const unsigned long long assumed = old;
        old = atomicCAS(address, assumed, desired);
        if (old == assumed) {
            break;
        }
    }
}

// Map current radius to the resolved mesh cell. The caller separately
// handles particles inside the first face and validates the outer domain.
__device__ inline int resolved_cell(
    double radius,
    const double* edge,
    int cells) {
    int first = 0;
    int last = cells + 1;
    while (first < last) {
        const int middle = first + (last - first) / 2;
        if (radius < edge[middle]) {
            last = middle;
        } else {
            first = middle + 1;
        }
    }
    return min(cells - 1, max(0, first - 1));
}

// Split the unresolved inner sphere into equal-volume radial cells;
// resolved cells retain the mesh geometry used by the gravitational source.
__device__ inline int physical_cell(
    double radius,
    const double* edge,
    int radial_cells,
    int inner_cells) {
    if (radius < edge[0]) {
        const double fraction = fmin(
            nextafter(1.0, 0.0),
            fmax(0.0, pow(radius / edge[0], 3.0)));
        return min(inner_cells - 1, static_cast<int>(fraction * inner_cells));
    }
    return inner_cells + resolved_cell(radius, edge, radial_cells);
}

// Assign active particles from their current radii. A deterministic ID/epoch
// hash distributes each physical population among sampling lanes without
// changing the physical-cell diagnostic grouping.
__global__ void assign_cell_keys_kernel(
    MarkerDeviceView markers,
    const std::uint64_t* active_order,
    std::uint64_t active_count,
    const double* edge,
    int radial_cells,
    DeviceSubhaloOptions options,
    std::uint64_t epoch,
    ProposalKey* proposal_key,
    std::uint64_t* physical_key,
    int* error_flag) {
    const std::uint64_t local =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local >= active_count) {
        return;
    }
    const std::uint64_t marker = active_order[local];
    const double radius = markers.r[marker];
    if (!(radius >= 0.0) || radius > edge[radial_cells] || !isfinite(radius)) {
        atomicMax(error_flag, 1);
        return;
    }
    if (radius == 0.0 && markers.eta[marker] != 0.0) {
        atomicMax(error_flag, 3);
        return;
    }
    // Match the CPU unequal-Plummer precondition before any majorant is built.
    // A zero dispersion would otherwise make both the pair scale and proposal
    // majorant vanish, silently turning a requested collisional update into a
    // no-op.  The energy relation is also needed for capture-heating closure.
    const double dispersion = markers.internal_velocity_dispersion[marker];
    const double internal_energy = markers.internal_energy[marker];
    const double expected_internal_energy =
        1.5 * markers.mass[marker] * dispersion * dispersion;
    const double internal_scale = fmax(
        1.0e-300, fmax(fabs(expected_internal_energy), fabs(internal_energy)));
    if (!(dispersion > 0.0) || !isfinite(dispersion)
        || !isfinite(internal_energy)
        || fabs(internal_energy - expected_internal_energy)
            > 2048.0 * 2.2204460492503131e-16 * internal_scale) {
        atomicMax(error_flag, 2);
        return;
    }
    const int physical = physical_cell(
        radius, edge, radial_cells, options.inner_cells);
    // Lane assignment is renewed for each batch epoch. Each lane receives
    // its fraction of cell volume so its sampled pair rates retain normalization.
    const std::uint64_t lane_hash = mix64(
        markers.id[marker]
        ^ mix64(epoch + UINT64_C(0x6a09e667f3bcc909)));
    const int lane = static_cast<int>(
        lane_hash % static_cast<std::uint64_t>(options.lanes));
    if (physical_key != nullptr) {
        physical_key[local] = static_cast<std::uint64_t>(physical);
    }
    if (proposal_key != nullptr) {
        proposal_key[local] = {
            static_cast<std::uint64_t>(physical * options.lanes + lane),
            markers.internal_velocity_dispersion[marker],
            markers.id[marker],
        };
    }
}

// Extract the sorted cell/lane component for lower_bound offset searches.
// The dispersion and ID components still determine order within each cell.
__global__ void extract_collision_keys_kernel(
    const ProposalKey* proposal_key,
    std::uint64_t count,
    std::uint64_t* collision_key) {
    const std::uint64_t local =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local < count) {
        collision_key[local] = proposal_key[local].cell;
    }
}

// Supply the volume used by the pair rate and the physical radial width
// used by locality diagnostics. Sampling lanes divide volume, not width.
__device__ inline void cell_geometry(
    int collision_cell,
    const double* edge,
    const double* shell_volume,
    DeviceSubhaloOptions options,
    double* volume,
    double* width) {
    const int physical = collision_cell / options.lanes;
    if (physical < options.inner_cells) {
        const double left = edge[0]
            * cbrt(static_cast<double>(physical) / options.inner_cells);
        const double right = edge[0]
            * cbrt(static_cast<double>(physical + 1) / options.inner_cells);
        // The CPU formula assigns identical volume to all virtual inner cells.
        // Do not recover it by subtracting two rounded cube roots: that tiny
        // difference can move a stochastic trial count across an integer.
        *volume = 4.0 * pi * edge[0] * edge[0] * edge[0]
            / (3.0 * options.inner_cells * options.lanes);
        *width = right - left;
    } else {
        const int resolved = physical - options.inner_cells;
        *volume = shell_volume[resolved] / options.lanes;
        *width = edge[resolved + 1] - edge[resolved];
    }
}

// Build the CPU pair-majorant proposal in one deterministic thread per cell.
// Cells are packed across ordinary CUDA blocks so all lanes in a warp can work
// on different cells.  Each individual cell still executes its prefix sums in
// the original serial order, preserving the CPU floating-point/RNG contract.
// The order array is already sorted by (cell, sigma_internal, marker_id).
// For i<j, sigma_j=max(sigma_i,sigma_j), and
//   B_ij = C sigma_j (sqrt(m_i)+sqrt(m_j))^2.
// Prefix sums make both the O(N) construction and conditional pair draw exact
// copies of the CPU algorithm without materializing O(N^2) pairs.
__global__ void build_pair_proposals_kernel(
    MarkerDeviceView markers,
    const std::uint64_t* order,
    const std::uint64_t* offsets,
    int cell_count,
    const double* edge,
    const double* shell_volume,
    DeviceSubhaloOptions options,
    DeviceSubhaloTable table,
    double* prefix_root,
    double* prefix_mass,
    double* pair_cumulative,
    DeviceCell* cells) {
    const int cell_index = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (cell_index >= cell_count) {
        return;
    }
    const std::uint64_t begin = offsets[cell_index];
    const std::uint64_t end = offsets[cell_index + 1];
    DeviceCell result{};
    result.begin = begin;
    result.end = end;
    cell_geometry(
        cell_index, edge, shell_volume, options, &result.volume, &result.width);
    const std::uint64_t population = end - begin;
    if (population < options.minimum_population) {
        cells[cell_index] = result;
        return;
    }

    // For each higher-dispersion partner j, sum all i<j pair weights using
    // sums of sqrt(m_i), m_i, and the partner count. This constructs the
    // exact cumulative majorant in linear work instead of storing all pairs.
    double running_root = 0.0;
    double running_mass = 0.0;
    double pair_weight_sum = 0.0;
    for (std::uint64_t local = 0; local < population; ++local) {
        const std::uint64_t position = begin + local;
        const std::uint64_t marker = order[position];
        const double root_mass = sqrt(markers.mass[marker]);
        prefix_root[position] = running_root;
        prefix_mass[position] = running_mass;
        const double lower_area_sum = running_mass
            + 2.0 * root_mass * running_root
            + static_cast<double>(local) * root_mass * root_mass;
        pair_weight_sum += markers.internal_velocity_dispersion[marker]
            * lower_area_sum;
        pair_cumulative[position] = pair_weight_sum;
        running_root += root_mass;
        running_mass += markers.mass[marker];
    }

    // The table bounds x*shape(x). Multiplication by a pair speed scale
    // no larger than max(sigma1,sigma2) bounds the physical u*Sigma kernel.
    const double bound =
        (options.enable_scattering != 0 ? table.scatter_bound : 0.0)
        + (options.enable_capture != 0 ? table.capture_bound : 0.0);
    const double pair_kernel_scale = 0.25 * options.strength
        * options.area_per_mass * bound
        * (1.0 + 128.0 * 2.2204460492503131e-16);
    result.pair_majorant = pair_kernel_scale * pair_weight_sum;

    // A backward scan adds partners above each marker in dispersion order.
    // Combined lower/upper sums bound that marker's total trial frequency,
    // which sets the local collision CFL independently of total trial count.
    double suffix_sigma = 0.0;
    double suffix_sigma_root = 0.0;
    double suffix_sigma_mass = 0.0;
    double maximum_marker_bound = 0.0;
    for (std::uint64_t reverse = population; reverse-- > 0;) {
        const std::uint64_t position = begin + reverse;
        const std::uint64_t marker = order[position];
        const double mass = markers.mass[marker];
        const double root_mass = sqrt(mass);
        const double sigma = markers.internal_velocity_dispersion[marker];
        const double lower = sigma
            * (prefix_mass[position]
               + 2.0 * root_mass * prefix_root[position]
               + static_cast<double>(reverse) * root_mass * root_mass);
        const double upper = root_mass * root_mass * suffix_sigma
            + 2.0 * root_mass * suffix_sigma_root + suffix_sigma_mass;
        maximum_marker_bound = fmax(maximum_marker_bound, lower + upper);
        suffix_sigma += sigma;
        suffix_sigma_root += sigma * root_mass;
        suffix_sigma_mass += sigma * mass;
    }
    result.maximum_trial_frequency =
        pair_kernel_scale * maximum_marker_bound / result.volume;
    cells[cell_index] = result;
}

// Measure physical-cell mean free path and encounter time from actual pair
// speeds/cross sections. Use all pairs for a small population and a bounded
// random pair sample otherwise, before applying artificial sampling lanes.
__global__ void diagnose_physical_cells_kernel(
    MarkerDeviceView markers,
    const std::uint64_t* order,
    const std::uint64_t* offsets,
    int physical_cell_count,
    const double* edge,
    const double* shell_volume,
    DeviceSubhaloOptions options,
    DeviceSubhaloTable table,
    std::uint64_t seed,
    std::uint64_t epoch,
    DevicePhysicalDiagnostic* diagnostics) {
    const int physical = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (physical >= physical_cell_count) {
        return;
    }
    const std::uint64_t begin = offsets[physical];
    const std::uint64_t end = offsets[physical + 1];
    const std::uint64_t population = end - begin;
    DevicePhysicalDiagnostic output{};
    output.mean_free_path = INFINITY;
    output.collision_timescale = INFINITY;
    output.minimum_pair_velocity_scale = INFINITY;
    if (population < options.minimum_population) {
        diagnostics[physical] = output;
        return;
    }
    // The central region receives the larger bounded sample. Pair extrema
    // reported below refer to these sampled pairs, rather than all angles.
    const std::uint64_t requested = physical < options.inner_cells
        ? UINT64_C(4096) : UINT64_C(256);
    const std::uint64_t pair_count = population * (population - 1) / 2;
    const std::uint64_t samples = min(requested, pair_count);
    double speed_sum = 0.0;
    double kernel_sum = 0.0;
    auto sample_pair = [&](std::uint64_t first_local,
                           std::uint64_t second_local,
                           std::uint64_t sample) {
        const std::uint64_t first = order[begin + first_local];
        const std::uint64_t second = order[begin + second_local];
        const double relative_azimuth = 2.0 * pi
            * subhalo_uniform(
                seed,
                epoch,
                static_cast<std::uint64_t>(physical)
                    + options.resolved_cell_offset,
                UINT64_C(0x6d66705f6466),
                sample,
                31);
        const double first_vt = sqrt(2.0 * markers.eta[first])
            / fmax(markers.r[first], 1.0e-300);
        const double second_vt = sqrt(2.0 * markers.eta[second])
            / fmax(markers.r[second], 1.0e-300);
        const double radial_difference = markers.w[first] - markers.w[second];
        double u = sqrt(fmax(
            0.0,
            radial_difference * radial_difference
                + first_vt * first_vt + second_vt * second_vt
                - 2.0 * first_vt * second_vt * cos(relative_azimuth)));
        if (markers.r[first] == 0.0 || markers.r[second] == 0.0) {
            const std::uint64_t global_cell =
                static_cast<std::uint64_t>(physical) + options.resolved_cell_offset;
            const auto first_velocity = sample_origin_velocity(
                markers.r[first], markers.w[first], markers.eta[first], 0.0,
                seed, epoch, global_cell, UINT64_C(0x6d66705f6466), sample, 34);
            const auto second_velocity = sample_origin_velocity(
                markers.r[second], markers.w[second], markers.eta[second],
                relative_azimuth, seed, epoch, global_cell,
                UINT64_C(0x6d66705f6466), sample, 36);
            u = origin_pair_speed(first_velocity, second_velocity);
        }
        const SubhaloCrossSections cross = evaluate_pair(
            u,
            markers.mass[first],
            markers.mass[second],
            markers.internal_velocity_dispersion[first],
            markers.internal_velocity_dispersion[second],
            options,
            table);
        speed_sum += u;
        kernel_sum += u * (cross.scattering + cross.capture);
        output.minimum_pair_velocity_scale = fmin(
            output.minimum_pair_velocity_scale,
            cross.pair_velocity_scale);
        output.maximum_pair_velocity_scale = fmax(
            output.maximum_pair_velocity_scale,
            cross.pair_velocity_scale);
    };
    // Enumerate each unordered pair once when affordable. Otherwise draw
    // distinct partners uniformly; the second index skips the selected first.
    if (pair_count <= requested) {
        std::uint64_t sample = 0;
        for (std::uint64_t first = 0; first < population; ++first) {
            for (std::uint64_t second = first + 1;
                 second < population;
                 ++second) {
                sample_pair(first, second, sample++);
            }
        }
    } else {
        for (std::uint64_t sample = 0; sample < samples; ++sample) {
            const std::uint64_t first = min(
                population - 1,
                static_cast<std::uint64_t>(
                    subhalo_uniform(
                        seed,
                        epoch,
                        static_cast<std::uint64_t>(physical)
                            + options.resolved_cell_offset,
                        UINT64_C(0x6d66705f6466),
                        sample,
                        32)
                    * static_cast<double>(population)));
            std::uint64_t second = min(
                population - 2,
                static_cast<std::uint64_t>(
                    subhalo_uniform(
                        seed,
                        epoch,
                        static_cast<std::uint64_t>(physical)
                            + options.resolved_cell_offset,
                        UINT64_C(0x6d66705f6466),
                        sample,
                        33)
                    * static_cast<double>(population - 1)));
            if (second >= first) {
                ++second;
            }
            sample_pair(first, second, sample);
        }
    }
    double volume = 0.0;
    double width = 0.0;
    cell_geometry(
        physical * options.lanes,
        edge,
        shell_volume,
        options,
        &volume,
        &width);
    volume *= options.lanes;
    // Restore the full physical volume after asking the lane geometry
    // helper. Mean speed divided by rate gives the physical path estimate.
    const double partner_density = static_cast<double>(population - 1) / volume;
    if (kernel_sum > 0.0) {
        output.mean_free_path = speed_sum
            / (partner_density * kernel_sum);
        output.collision_timescale = static_cast<double>(samples)
            / (partner_density * kernel_sum);
        output.width_over_mean_free_path = width / output.mean_free_path;
        output.nonlocal = output.width_over_mean_free_path > 1.0 ? 1 : 0;
    }
    output.pair_samples = samples;
    diagnostics[physical] = output;
}

// Advance one collision cell through one local collision substep.  Capture
// changes mass and internal dispersion, so the return value tells the host to
// rebuild the proposal before that cell advances again.  Elastic scattering
// leaves every proposal input unchanged.
__device__ bool collide_subhalo_cell_substep(
    MarkerDeviceView markers,
    const std::uint64_t* order,
    const DeviceCell* cells,
    const double* prefix_root,
    const double* prefix_mass,
    const double* pair_cumulative,
    int cell_index,
    DeviceSubhaloOptions options,
    DeviceSubhaloTable table,
    double subdt,
    std::uint64_t seed,
    std::uint64_t epoch,
    std::uint64_t substep,
    std::uint8_t* collided,
    std::uint64_t* collided_generation,
    std::uint8_t* remove_marker,
    DeviceSubhaloStats* stats) {
    bool captured_in_cell = false;
    const DeviceCell cell = cells[cell_index];
    const std::uint64_t population = cell.end - cell.begin;
    if (population < options.minimum_population) {
        return false;
    }
    // Trial rate is the sum of pair majorants divided by cell volume.
    // Randomized rounding preserves its expectation for a noninteger count.
    const double expected_trials = cell.pair_majorant * subdt / cell.volume;
    if (!(expected_trials >= 0.0)
        || expected_trials
            > static_cast<double>(UINT64_MAX - UINT64_C(1))) {
        atomicExch(&stats->error_flag, 1);
        return false;
    }
    const double rounding = subhalo_uniform(
        seed,
        epoch,
        static_cast<std::uint64_t>(cell_index)
            + options.resolved_cell_offset
                * static_cast<std::uint64_t>(options.lanes),
        substep,
        UINT64_MAX,
        0);
    const std::uint64_t trials = static_cast<std::uint64_t>(
        floor(expected_trials + rounding));
    for (std::uint64_t trial = 0; trial < trials; ++trial) {
        atomicAdd(&stats->trial_pairs, UINT64_C(1));
        // First choose partner j from the cumulative sum of its i<j
        // majorants, then sample partner i conditionally from that same weight.
        const double group_draw = subhalo_uniform(
            seed,
            epoch,
            static_cast<std::uint64_t>(cell_index)
                + options.resolved_cell_offset
                    * static_cast<std::uint64_t>(options.lanes),
            substep,
            trial,
            1) * pair_cumulative[cell.end - 1];
        const int group_offset = upper_index(
            pair_cumulative + cell.begin + 1,
            static_cast<int>(population - 1),
            group_draw);
        const std::uint64_t second_local = min(
            population - 1,
            static_cast<std::uint64_t>(group_offset + 1));
        const std::uint64_t high_position = cell.begin + second_local;
        const std::uint64_t high_marker = order[high_position];
        const double high_root = sqrt(markers.mass[high_marker]);
        // Expand (sqrt(m_i)+sqrt(m_j))^2 into an m_i term, a sqrt(m_i)
        // term, and a constant. Three prefix/uniform draws sample those
        // components exactly without an all-pairs table.
        const double mass_component = prefix_mass[high_position];
        const double root_component =
            2.0 * high_root * prefix_root[high_position];
        const double uniform_component = static_cast<double>(second_local)
            * high_root * high_root;
        double conditional_draw = subhalo_uniform(
            seed,
            epoch,
            static_cast<std::uint64_t>(cell_index)
                + options.resolved_cell_offset
                    * static_cast<std::uint64_t>(options.lanes),
            substep,
            trial,
            2) * (mass_component + root_component + uniform_component);
        std::uint64_t first_local = 0;
        if (conditional_draw < mass_component) {
            first_local = static_cast<std::uint64_t>(upper_index(
                prefix_mass + cell.begin + 1,
                static_cast<int>(second_local),
                conditional_draw));
        } else if ((conditional_draw -= mass_component) < root_component) {
            const double root_draw = conditional_draw / (2.0 * high_root);
            first_local = static_cast<std::uint64_t>(upper_index(
                prefix_root + cell.begin + 1,
                static_cast<int>(second_local),
                root_draw));
        } else {
            conditional_draw -= root_component;
            first_local = min(
                second_local - 1,
                static_cast<std::uint64_t>(
                    conditional_draw / (high_root * high_root)));
        }
        const std::uint64_t first_position = cell.begin + first_local;
        const std::uint64_t second_position = cell.begin + second_local;
        // A particle participates in at most one accepted encounter per
        // local substep. Generation tags implement this across batched local
        // steps; the reference path instead clears a byte mask each step.
        const std::uint64_t generation = substep + UINT64_C(1);
        const bool first_collided = collided_generation != nullptr
            ? collided_generation[first_position] == generation
            : collided[first_position] != 0;
        const bool second_collided = collided_generation != nullptr
            ? collided_generation[second_position] == generation
            : collided[second_position] != 0;
        if (first_collided || second_collided) {
            atomicAdd(
                &stats->skipped_already_collided_trials, UINT64_C(1));
            continue;
        }
        const std::uint64_t first = order[cell.begin + first_local];
        const std::uint64_t second = order[cell.begin + second_local];
        // Capture products remain in storage until stable compaction.
        // A removed partner is excluded from every later trial in the batch.
        if (remove_marker[first] != 0 || remove_marker[second] != 0) {
            continue;
        }
        const double relative_azimuth = 2.0 * pi
            * subhalo_uniform(
                seed,
                epoch,
                static_cast<std::uint64_t>(cell_index)
                    + options.resolved_cell_offset
                        * static_cast<std::uint64_t>(options.lanes),
                substep,
                trial,
                3);
        const double first_radius = fmax(markers.r[first], 1.0e-300);
        const double second_radius = fmax(markers.r[second], 1.0e-300);
        const double first_vt = sqrt(2.0 * markers.eta[first]) / first_radius;
        const double second_vt = sqrt(2.0 * markers.eta[second]) / second_radius;
        const double radial_difference = markers.w[first] - markers.w[second];
        double relative_speed = sqrt(fmax(
            0.0,
            radial_difference * radial_difference
                + first_vt * first_vt + second_vt * second_vt
                - 2.0 * first_vt * second_vt * cos(relative_azimuth)));
        // At the exact origin only speed is retained by spherical symmetry.
        // Reuse the same isotropic vectors for the rate and outgoing map.
        const bool origin_pair = markers.r[first] == 0.0 || markers.r[second] == 0.0;
        OriginPairVelocity origin_first{}, origin_second{};
        if (origin_pair) {
            const std::uint64_t global_cell = static_cast<std::uint64_t>(cell_index)
                + options.resolved_cell_offset * static_cast<std::uint64_t>(options.lanes);
            const double azimuth = 2.0 * pi * subhalo_uniform(
                seed, epoch, global_cell, substep, trial, 4);
            origin_first = sample_origin_velocity(
                markers.r[first], markers.w[first], markers.eta[first], azimuth,
                seed, epoch, global_cell, substep, trial, 10);
            origin_second = sample_origin_velocity(
                markers.r[second], markers.w[second], markers.eta[second],
                azimuth + relative_azimuth, seed, epoch, global_cell, substep, trial, 12);
            relative_speed = origin_pair_speed(origin_first, origin_second);
        }
        const double first_mass = markers.mass[first];
        const double second_mass = markers.mass[second];
        const SubhaloCrossSections cross = evaluate_pair(
            relative_speed,
            first_mass,
            second_mass,
            markers.internal_velocity_dispersion[first],
            markers.internal_velocity_dispersion[second],
            options,
            table);
        // Accept the physical summed channel kernel divided by its pair
        // majorant. A value outside the roundoff allowance rejects the batch.
        double probability = relative_speed
            * (cross.scattering + cross.capture) / cross.u_sigma_majorant;
        if (probability > 1.0 + 1024.0 * 2.2204460492503131e-16) {
            atomicExch(&stats->error_flag, 1);
            continue;
        }
        probability = fmin(1.0, fmax(0.0, probability));
        if (subhalo_uniform(
                seed,
                epoch,
                static_cast<std::uint64_t>(cell_index)
                    + options.resolved_cell_offset
                        * static_cast<std::uint64_t>(options.lanes),
                substep,
                trial,
                5)
            >= probability) {
            continue;
        }
        // Set the one-encounter mark only after accepting the physical pair,
        // allowing rejected majorant proposals to leave both particles eligible.
        if (collided_generation != nullptr) {
            collided_generation[first_position] = generation;
            collided_generation[second_position] = generation;
        } else {
            collided[first_position] = 1;
            collided[second_position] = 1;
        }
        atomicAdd(&stats->accepted_pairs, UINT64_C(1));

        const double first_azimuth = 2.0 * pi
            * subhalo_uniform(
                seed,
                epoch,
                static_cast<std::uint64_t>(cell_index)
                    + options.resolved_cell_offset
                        * static_cast<std::uint64_t>(options.lanes),
                substep,
                trial,
                4);
        const double second_azimuth = first_azimuth + relative_azimuth;
        const double v1x = origin_pair ? origin_first.x : markers.w[first];
        const double v1y = origin_pair ? origin_first.y : first_vt * cos(first_azimuth);
        const double v1z = origin_pair ? origin_first.z : first_vt * sin(first_azimuth);
        const double v2x = origin_pair ? origin_second.x : markers.w[second];
        const double v2y = origin_pair ? origin_second.y : second_vt * cos(second_azimuth);
        const double v2z = origin_pair ? origin_second.z : second_vt * sin(second_azimuth);

        // Compute center-of-mass and relative kinetic energies before
        // choosing capture versus scattering. Their sum supplies the energy
        // budget used to check the outgoing physical map.
        const double pair_mass = first_mass + second_mass;
        const double reduced_mass = first_mass * second_mass / pair_mass;
        const double px = first_mass * v1x + second_mass * v2x;
        const double py = first_mass * v1y + second_mass * v2y;
        const double pz = first_mass * v1z + second_mass * v2z;
        const double center_x = px / pair_mass;
        const double center_y = py / pair_mass;
        const double center_z = pz / pair_mass;
        const double kinetic_before = 0.5
            * (first_mass * (v1x * v1x + v1y * v1y + v1z * v1z)
               + second_mass * (v2x * v2x + v2y * v2y + v2z * v2z));
        const double internal_before = markers.internal_energy[first]
            + markers.internal_energy[second];
        // Conditional channel selection uses capture/(scatter+capture),
        // consistent with the total rate already used for acceptance.
        const bool capture = cross.capture > 0.0
            && subhalo_uniform(
                    seed,
                    epoch,
                    static_cast<std::uint64_t>(cell_index)
                        + options.resolved_cell_offset
                            * static_cast<std::uint64_t>(options.lanes),
                    substep,
                    trial,
                    6)
                < cross.capture / (cross.scattering + cross.capture);
        if (capture) {
            captured_in_cell = true;
            // Capture removes the relative kinetic energy from resolved
            // orbital motion and adds it to the combined internal reservoir.
            // The survivor carries total mass at the mass-weighted radius.
            const double lost_orbital =
                0.5 * reduced_mass * relative_speed * relative_speed;
            const double merged_internal = internal_before + lost_orbital;
            const double merged_radius =
                (first_mass * markers.r[first]
                 + second_mass * markers.r[second]) / pair_mass;
            markers.mass[first] = pair_mass;
            markers.r[first] = merged_radius;
            markers.w[first] = center_x;
            markers.eta[first] = 0.5 * merged_radius * merged_radius
                * (center_y * center_y + center_z * center_z);
            // A remnant exactly at the origin stores center-of-mass speed
            // with eta=0, the regular spherical representation at that point.
            if (merged_radius == 0.0) {
                markers.w[first] = hypot(center_x, hypot(center_y, center_z));
                markers.eta[first] = 0.0;
            }
            markers.internal_energy[first] = merged_internal;
            // Recover the remnant dispersion from E_internal=3*M*sigma^2/2
            // so its next proposal and later capture heating use one state.
            markers.internal_velocity_dispersion[first] = sqrt(
                fmax(0.0, 2.0 * merged_internal / (3.0 * pair_mass)));
            remove_marker[second] = 1;
            atomicAdd(&stats->capture_events, UINT64_C(1));
            atomicAdd(&stats->removed_markers, UINT64_C(1));
            if (cell_index < options.inner_cells * options.lanes) {
                atomicAdd(&stats->inner_capture_events, UINT64_C(1));
            }
            atomicAdd(&stats->represented_capture_mass, pair_mass);
            const double kinetic_after = 0.5 * pair_mass
                * (center_x * center_x + center_y * center_y
                   + center_z * center_z);
            atomicAdd(
                &stats->orbital_energy_change, kinetic_after - kinetic_before);
            atomicAdd(&stats->internal_energy_change, lost_orbital);
            // Close total orbital plus internal energy and vector momentum
            // for this actual capture outcome before removing the partner.
            const double energy_error = fabs(
                kinetic_after + merged_internal
                - kinetic_before - internal_before);
            atomic_max_bits(&stats->max_pair_energy_bits, energy_error);
            const double after_px = pair_mass * center_x;
            const double after_py = pair_mass * center_y;
            const double after_pz = pair_mass * center_z;
            const double momentum_error = sqrt(
                (after_px - px) * (after_px - px)
                + (after_py - py) * (after_py - py)
                + (after_pz - pz) * (after_pz - pz));
            atomic_max_bits(&stats->max_pair_momentum_bits, momentum_error);
            continue;
        }

        // The calibrated total scattering rate uses an isotropic outgoing
        // relative direction in this operator. Keep its speed fixed and split
        // the result around the unchanged center-of-mass velocity.
        const double mu = 2.0 * subhalo_uniform(
            seed,
            epoch,
            static_cast<std::uint64_t>(cell_index)
                + options.resolved_cell_offset
                    * static_cast<std::uint64_t>(options.lanes),
            substep,
            trial,
            7) - 1.0;
        const double azimuth = 2.0 * pi
            * subhalo_uniform(
                seed,
                epoch,
                static_cast<std::uint64_t>(cell_index)
                    + options.resolved_cell_offset
                        * static_cast<std::uint64_t>(options.lanes),
                substep,
                trial,
                8);
        const double sine = sqrt(fmax(0.0, 1.0 - mu * mu));
        const double rx = relative_speed * mu;
        const double ry = relative_speed * sine * cos(azimuth);
        const double rz = relative_speed * sine * sin(azimuth);
        const double first_fraction = second_mass / pair_mass;
        const double second_fraction = first_mass / pair_mass;
        const double out1x = center_x + first_fraction * rx;
        const double out1y = center_y + first_fraction * ry;
        const double out1z = center_z + first_fraction * rz;
        const double out2x = center_x - second_fraction * rx;
        const double out2y = center_y - second_fraction * ry;
        const double out2z = center_z - second_fraction * rz;
        markers.w[first] = out1x;
        markers.eta[first] = 0.5 * first_radius * first_radius
            * (out1y * out1y + out1z * out1z);
        markers.w[second] = out2x;
        markers.eta[second] = 0.5 * second_radius * second_radius
            * (out2y * out2y + out2z * out2z);
        if (markers.r[first] == 0.0) {
            markers.w[first] = hypot(out1x, hypot(out1y, out1z));
            markers.eta[first] = 0.0;
        }
        if (markers.r[second] == 0.0) {
            markers.w[second] = hypot(out2x, hypot(out2y, out2z));
            markers.eta[second] = 0.0;
        }
        atomicAdd(&stats->scattering_events, UINT64_C(1));
        if (cell_index < options.inner_cells * options.lanes) {
            atomicAdd(&stats->inner_scattering_events, UINT64_C(1));
        }
        atomicAdd(&stats->represented_scattering_mass, pair_mass);
        // Scattering retains each mass and internal energy. Recompute pair
        // momentum and kinetic energy from the outgoing velocities as checks.
        const double kinetic_after = 0.5
            * (first_mass * (out1x * out1x + out1y * out1y + out1z * out1z)
               + second_mass
                   * (out2x * out2x + out2y * out2y + out2z * out2z));
        atomicAdd(
            &stats->orbital_energy_change, kinetic_after - kinetic_before);
        const double after_px = first_mass * out1x + second_mass * out2x;
        const double after_py = first_mass * out1y + second_mass * out2y;
        const double after_pz = first_mass * out1z + second_mass * out2z;
        const double momentum_error = sqrt(
            (after_px - px) * (after_px - px)
            + (after_py - py) * (after_py - py)
            + (after_pz - pz) * (after_pz - pz));
        atomic_max_bits(&stats->max_pair_momentum_bits, momentum_error);
        atomic_max_bits(
            &stats->max_pair_energy_bits, fabs(kinetic_after - kinetic_before));
    }
    return captured_in_cell;
}

// Reference operator: the fastest cell supplies one mesh-wide collision
// substep and every eligible cell is visited once per launch.
__global__ void collide_subhalo_cells_kernel(
    MarkerDeviceView markers,
    const std::uint64_t* order,
    const DeviceCell* cells,
    const double* prefix_root,
    const double* prefix_mass,
    const double* pair_cumulative,
    int cell_count,
    DeviceSubhaloOptions options,
    DeviceSubhaloTable table,
    double subdt,
    std::uint64_t seed,
    std::uint64_t epoch,
    std::uint64_t substep,
    std::uint8_t* collided,
    std::uint8_t* remove_marker,
    DeviceSubhaloStats* stats) {
    const int cell_index = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (cell_index >= cell_count) {
        return;
    }
    collide_subhalo_cell_substep(
        markers,
        order,
        cells,
        prefix_root,
        prefix_mass,
        pair_cumulative,
        cell_index,
        options,
        table,
        subdt,
        seed,
        epoch,
        substep,
        collided,
        nullptr,
        remove_marker,
        stats);
}

// Adaptive operator: one CUDA thread owns one collision cell and advances its
// independent clock through several local CFL substeps.  Capture terminates
// that cell's batch immediately, preventing reuse of a stale pair majorant.
__global__ void collide_subhalo_cells_adaptive_kernel(
    MarkerDeviceView markers,
    const std::uint64_t* order,
    const DeviceCell* cells,
    const double* prefix_root,
    const double* prefix_mass,
    const double* pair_cumulative,
    int cell_count,
    DeviceSubhaloOptions options,
    DeviceSubhaloTable table,
    double interval,
    std::uint64_t maximum_substeps,
    int batch_substeps,
    std::uint64_t seed,
    std::uint64_t epoch,
    double* cell_elapsed,
    std::uint64_t* cell_substeps,
    std::uint64_t* collided_generation,
    std::uint8_t* remove_marker,
    DeviceSubhaloStats* stats) {
    const int cell_index = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (cell_index >= cell_count) {
        return;
    }
    const DeviceCell cell = cells[cell_index];
    const std::uint64_t population = cell.end - cell.begin;
    if (population < options.minimum_population) {
        return;
    }

    // Local elapsed time and substep count persist across kernel batches.
    // Capture returns early so changed masses/dispersions are sorted and
    // rebuilt before this same cell consumes any further physical time.
    double elapsed = cell_elapsed[cell_index];
    std::uint64_t substep = cell_substeps[cell_index];
    int batched = 0;
    while (elapsed < interval && batched < batch_substeps) {
        if (substep >= maximum_substeps) {
            atomicExch(&stats->error_flag, 2);
            break;
        }
        const double remaining = interval - elapsed;
        const double subdt = cell.maximum_trial_frequency > 0.0
            ? fmin(
                  remaining,
                  options.collision_cfl / cell.maximum_trial_frequency)
            : remaining;
        if (!(subdt > 0.0) || !isfinite(subdt)) {
            atomicExch(&stats->error_flag, 1);
            break;
        }
        const bool captured = collide_subhalo_cell_substep(
            markers,
            order,
            cells,
            prefix_root,
            prefix_mass,
            pair_cumulative,
            cell_index,
            options,
            table,
            subdt,
            seed,
            epoch,
            substep,
            nullptr,
            collided_generation,
            remove_marker,
            stats);
        // Consume the exact remaining interval at the final local step,
        // preventing a tiny arithmetic remainder from generating another step.
        elapsed = subdt >= remaining ? interval : elapsed + subdt;
        ++substep;
        ++batched;
        atomicAdd(&stats->cell_visits, UINT64_C(1));
        if (captured) {
            break;
        }
    }
    cell_elapsed[cell_index] = elapsed;
    cell_substeps[cell_index] = substep;
    atomicMax(&stats->maximum_cell_substeps, substep);
    if (elapsed < interval) {
        atomicAdd(&stats->unfinished_cells, UINT64_C(1));
    }
}

// Validate calibration range, positive area/dispersion controls, finite
// strength, collision CFL, work bounds, and indexing before launching kernels.
DeviceSubhaloOptions make_device_options(
    const SubhaloCollisionOptions& options) {
    if (!(options.velocity_scale > 0.0)
        || !std::isfinite(options.velocity_scale)
        || !(options.plummer_area_per_mass > 0.0)
        || !std::isfinite(options.plummer_area_per_mass)
        || !(options.low_speed_regularization
             >= generated_subhalo_table::speed[0])
        || !(options.low_speed_regularization < 1.0)
        || !std::isfinite(options.low_speed_regularization)
        || !(options.strength >= 0.0) || !std::isfinite(options.strength)
        || !(options.collision_cfl > 0.0 && options.collision_cfl <= 1.0)
        || options.maximum_substeps < 1
        || options.inner_collision_cells < 1
        || options.inner_collision_cells > 256
        || options.collision_lanes < 1
        || options.collision_lanes > 256
        || options.adaptive_batch_substeps < 1
        || options.minimum_cell_population < 2
        || (options.strength > 0.0
            && !options.enable_scattering && !options.enable_capture)) {
        throw std::runtime_error("Invalid GPU subhalo collision options.");
    }
    DeviceSubhaloOptions result;
    result.area_per_mass = options.plummer_area_per_mass;
    result.low_speed_regularization = options.low_speed_regularization;
    result.strength = options.strength;
    result.collision_cfl = options.collision_cfl;
    result.inner_cells = options.inner_collision_cells;
    result.lanes = options.collision_lanes;
    result.minimum_population = options.minimum_cell_population;
    result.resolved_cell_offset = options.resolved_cell_offset;
    result.enable_scattering = options.enable_scattering ? 1 : 0;
    result.enable_capture = options.enable_capture ? 1 : 0;
    return result;
}

double bits_to_double(unsigned long long bits) {
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(double));
    return value;
}

}  // namespace

using detail::DeviceArray;

// Persist mesh, calibration tables, sorted indices, prefix sums, local
// clocks, and compact statistics. WpicState remains the physical-state owner.
struct SubhaloCollisionModule::Impl {
    explicit Impl(const RadialMesh& radial_mesh)
        : mesh(radial_mesh), stats(1), error_flag(1) {
        mesh_edge.upload(mesh.edge.data(), mesh.edge.size());
        shell_volume.upload(mesh.shell_volume.data(), mesh.shell_volume.size());
        table_mass_ratio.upload(
            generated_subhalo_table::mass_ratio,
            generated_subhalo_table::mass_ratio_count);
        table_speed.upload(
            generated_subhalo_table::speed,
            generated_subhalo_table::speed_count);
        table_capture.upload(
            generated_subhalo_table::capture,
            generated_subhalo_table::mass_ratio_count
                * generated_subhalo_table::speed_count);
        table_scatter.upload(
            generated_subhalo_table::scatter,
            generated_subhalo_table::mass_ratio_count
                * generated_subhalo_table::speed_count);
        table_scatter_high.upload(
            generated_subhalo_table::scatter_high_speed_coefficient,
            generated_subhalo_table::mass_ratio_count);
    }

    DeviceSubhaloTable device_table() const {
        return {
            table_mass_ratio.data(),
            table_speed.data(),
            table_capture.data(),
            table_scatter.data(),
            table_scatter_high.data(),
            generated_subhalo_table::capture_u_shape_bound,
            generated_subhalo_table::scatter_u_shape_bound,
        };
    }

    RadialMesh mesh;
    DeviceArray<double> mesh_edge;
    DeviceArray<double> shell_volume;
    DeviceArray<double> table_mass_ratio;
    DeviceArray<double> table_speed;
    DeviceArray<double> table_capture;
    DeviceArray<double> table_scatter;
    DeviceArray<double> table_scatter_high;
    DeviceArray<std::uint64_t> active_order;
    DeviceArray<std::uint64_t> physical_order;
    DeviceArray<ProposalKey> proposal_key;
    DeviceArray<std::uint64_t> collision_key;
    DeviceArray<std::uint64_t> physical_key;
    DeviceArray<std::uint64_t> collision_offsets;
    DeviceArray<std::uint64_t> physical_offsets;
    DeviceArray<DeviceCell> cells;
    DeviceArray<double> prefix_root;
    DeviceArray<double> prefix_mass;
    DeviceArray<double> pair_cumulative;
    DeviceArray<DevicePhysicalDiagnostic> physical_diagnostics;
    DeviceArray<std::uint8_t> collided;
    DeviceArray<std::uint64_t> collided_generation;
    DeviceArray<std::uint8_t> remove_marker;
    DeviceArray<double> cell_elapsed;
    DeviceArray<std::uint64_t> cell_substeps;
    DeviceArray<DeviceSubhaloStats> stats;
    DeviceArray<int> error_flag;
};

SubhaloCollisionModule::SubhaloCollisionModule(const RadialMesh& mesh)
    : impl_(new Impl(mesh)) {}

SubhaloCollisionModule::~SubhaloCollisionModule() = default;
SubhaloCollisionModule::SubhaloCollisionModule(
    SubhaloCollisionModule&&) noexcept = default;
SubhaloCollisionModule& SubhaloCollisionModule::operator=(
    SubhaloCollisionModule&&) noexcept = default;

// The public host evaluator uses the same checked pair law and table as
// CUDA, allowing pair rates to be inspected without constructing GPU state.
SubhaloCrossSections subhalo_cross_sections(
    double relative_speed,
    double first_mass,
    double second_mass,
    double first_internal_dispersion,
    double second_internal_dispersion,
    const SubhaloCollisionOptions& options) {
    const DeviceSubhaloOptions device_options = make_device_options(options);
    if (!(relative_speed >= 0.0) || !std::isfinite(relative_speed)
        || !(first_mass > 0.0) || !std::isfinite(first_mass)
        || !(second_mass > 0.0) || !std::isfinite(second_mass)
        || !(first_internal_dispersion > 0.0)
        || !std::isfinite(first_internal_dispersion)
        || !(second_internal_dispersion > 0.0)
        || !std::isfinite(second_internal_dispersion)) {
        throw std::runtime_error("Invalid unequal-mass subhalo pair.");
    }
    const DeviceSubhaloTable host_table{
        generated_subhalo_table::mass_ratio,
        generated_subhalo_table::speed,
        generated_subhalo_table::capture,
        generated_subhalo_table::scatter,
        generated_subhalo_table::scatter_high_speed_coefficient,
        generated_subhalo_table::capture_u_shape_bound,
        generated_subhalo_table::scatter_u_shape_bound,
    };
    return evaluate_pair(
        relative_speed,
        first_mass,
        second_mass,
        first_internal_dispersion,
        second_internal_dispersion,
        device_options,
        host_table);
}

// One outer collision batch takes one sampler epoch. Build current-radius
// orders, diagnose complete physical cells, then cover dt using common or
// independent local clocks while rebuilding proposals after every capture.
SubhaloCollisionDiagnostics SubhaloCollisionModule::apply(
    WpicState& state,
    double dt,
    const SubhaloCollisionOptions& options,
    CollisionSamplerState& sampler) {
    if (!(dt >= 0.0) || !std::isfinite(dt)) {
        throw std::runtime_error("Invalid GPU subhalo collision interval.");
    }
    const DeviceSubhaloOptions device_options = make_device_options(options);
    SubhaloCollisionDiagnostics total;
    if (dt == 0.0 || options.strength == 0.0
        || (state.size() < 2 && options.maximum_frequency_reducer == nullptr)) {
        return total;
    }
    if (sampler.next_counter == UINT64_MAX) {
        throw std::runtime_error(
            "GPU subhalo collision sampler counter exhausted.");
    }
    if (impl_->mesh.cells()
            > static_cast<std::size_t>(INT_MAX - options.inner_collision_cells)
        || impl_->mesh.cells()
                + static_cast<std::size_t>(options.inner_collision_cells)
            > static_cast<std::size_t>(
                INT_MAX / options.collision_lanes)) {
        throw std::runtime_error(
            "GPU subhalo collision cell count exceeds CUDA int indexing.");
    }
    const std::uint64_t physical_cell_count_u64 =
        static_cast<std::uint64_t>(impl_->mesh.cells())
        + static_cast<std::uint64_t>(options.inner_collision_cells);
    const std::uint64_t collision_lanes_u64 =
        static_cast<std::uint64_t>(options.collision_lanes);
    if (options.resolved_cell_offset
        > UINT64_MAX / collision_lanes_u64 - physical_cell_count_u64) {
        throw std::runtime_error(
            "GPU subhalo resolved-cell RNG offset overflows uint64_t.");
    }
    // A single batch epoch plus cell/substep/trial/dimension keys identifies
    // random draws independently of the order in which GPU cells are scheduled.
    const std::uint64_t epoch = sampler.next_counter++;
    const int physical_cell_count = options.inner_collision_cells
        + static_cast<int>(impl_->mesh.cells());
    const int cell_count = physical_cell_count * options.collision_lanes;
    const int physical_blocks =
        (physical_cell_count + subhalo_threads - 1) / subhalo_threads;
    const int cell_blocks =
        (cell_count + subhalo_threads - 1) / subhalo_threads;
    const DeviceSubhaloTable table = impl_->device_table();

    auto device_sum = [](const double* values, std::size_t count) {
        return thrust::reduce(
            thrust::device_pointer_cast(values),
            thrust::device_pointer_cast(values) + count,
            0.0,
            thrust::plus<double>());
    };
    const MarkerDeviceView initial_view = state.device_view();
    // Record global represented mass and internal energy before mutation.
    // Their final sums must close the capture and heating event ledgers.
    const double invariant_mass = device_sum(initial_view.mass, initial_view.size);
    const double initial_internal = device_sum(
        initial_view.internal_energy, initial_view.size);

    std::size_t active_count = 0;
    // Maintain two index views: dispersion-sorted cell/lanes for proposals
    // and whole physical cells for locality. Neither view reorders records.
    auto build_orders = [&]() {
        const MarkerDeviceView markers = state.device_view();
        impl_->active_order.resize(markers.size);
        const auto first = thrust::counting_iterator<std::uint64_t>(0);
        auto output = thrust::device_pointer_cast(impl_->active_order.data());
        const auto output_end = thrust::copy_if(
            first,
            first + static_cast<std::uint64_t>(markers.size),
            output,
            ActivePredicate{markers.state});
        active_count = static_cast<std::size_t>(output_end - output);
        impl_->physical_order.resize(active_count);
        impl_->proposal_key.resize(active_count);
        impl_->collision_key.resize(active_count);
        impl_->physical_key.resize(active_count);
        if (active_count == 0) {
            return;
        }
        VGPU_CUDA_CHECK(cudaMemcpy(
            impl_->physical_order.data(),
            impl_->active_order.data(),
            active_count * sizeof(std::uint64_t),
            cudaMemcpyDeviceToDevice));
        impl_->error_flag.zero();
        const int blocks = static_cast<int>(
            (active_count + subhalo_threads - 1) / subhalo_threads);
        assign_cell_keys_kernel<<<blocks, subhalo_threads>>>(
            markers,
            impl_->active_order.data(),
            active_count,
            impl_->mesh_edge.data(),
            static_cast<int>(impl_->mesh.cells()),
            device_options,
            epoch,
            impl_->proposal_key.data(),
            impl_->physical_key.data(),
            impl_->error_flag.data());
        detail::check_kernel("GPU subhalo cell assignment failed");
        int error = 0;
        impl_->error_flag.download(&error, 1);
        if (error == 3) {
            throw std::runtime_error(
                "A subhalo at the origin must have zero angular momentum.");
        }
        if (error == 1) {
            throw std::runtime_error(
                "Active subhalo marker lies outside the collision mesh.");
        }
        if (error != 0) {
            throw std::runtime_error(
                "GPU subhalo collision input requires every active marker "
                "to satisfy E_int=3*m*sigma_s^2/2 with sigma_s>0.");
        }
        // Increasing internal dispersion makes sigma_j the pair maximum
        // for i<j. ID supplies deterministic order for equal dispersions.
        thrust::sort_by_key(
            thrust::device_pointer_cast(impl_->proposal_key.data()),
            thrust::device_pointer_cast(impl_->proposal_key.data())
                + active_count,
            thrust::device_pointer_cast(impl_->active_order.data()),
            ProposalKeyLess{});
        thrust::stable_sort_by_key(
            thrust::device_pointer_cast(impl_->physical_key.data()),
            thrust::device_pointer_cast(impl_->physical_key.data())
                + active_count,
            thrust::device_pointer_cast(impl_->physical_order.data()));
        extract_collision_keys_kernel<<<blocks, subhalo_threads>>>(
            impl_->proposal_key.data(),
            active_count,
            impl_->collision_key.data());
        detail::check_kernel("GPU subhalo proposal-key extraction failed");
        impl_->collision_offsets.resize(static_cast<std::size_t>(cell_count + 1));
        impl_->physical_offsets.resize(
            static_cast<std::size_t>(physical_cell_count + 1));
        // Sorted-key lower bounds produce [begin,end) offsets, including
        // empty cells, for both proposal lanes and complete physical cells.
        thrust::lower_bound(
            thrust::device_pointer_cast(impl_->collision_key.data()),
            thrust::device_pointer_cast(impl_->collision_key.data())
                + active_count,
            thrust::counting_iterator<std::uint64_t>(0),
            thrust::counting_iterator<std::uint64_t>(
                static_cast<std::uint64_t>(cell_count + 1)),
            thrust::device_pointer_cast(impl_->collision_offsets.data()));
        thrust::lower_bound(
            thrust::device_pointer_cast(impl_->physical_key.data()),
            thrust::device_pointer_cast(impl_->physical_key.data())
                + active_count,
            thrust::counting_iterator<std::uint64_t>(0),
            thrust::counting_iterator<std::uint64_t>(
                static_cast<std::uint64_t>(physical_cell_count + 1)),
            thrust::device_pointer_cast(impl_->physical_offsets.data()));
    };

    build_orders();
    total.active_markers_before = active_count;
    if (active_count < 2 && options.maximum_frequency_reducer == nullptr) {
        total.active_markers_after = active_count;
        return total;
    }

    // Match the current CPU locality diagnostic: sample the complete physical
    // radial cell before splitting it into independent Monte-Carlo lanes.
    impl_->physical_diagnostics.resize(
        static_cast<std::size_t>(physical_cell_count));
    diagnose_physical_cells_kernel<<<physical_blocks, subhalo_threads>>>(
        state.device_view(),
        impl_->physical_order.data(),
        impl_->physical_offsets.data(),
        physical_cell_count,
        impl_->mesh_edge.data(),
        impl_->shell_volume.data(),
        device_options,
        table,
        sampler.seed,
        epoch,
        impl_->physical_diagnostics.data());
    detail::check_kernel("GPU subhalo locality diagnostic failed");
    std::vector<DevicePhysicalDiagnostic> physical(
        static_cast<std::size_t>(physical_cell_count));
    impl_->physical_diagnostics.download(physical.data(), physical.size());
    total.minimum_mean_free_path = std::numeric_limits<double>::infinity();
    total.minimum_collision_timescale = std::numeric_limits<double>::infinity();
    total.minimum_pair_velocity_scale =
        std::numeric_limits<double>::infinity();
    for (const auto& cell : physical) {
        total.minimum_mean_free_path = std::min(
            total.minimum_mean_free_path, cell.mean_free_path);
        total.minimum_collision_timescale = std::min(
            total.minimum_collision_timescale, cell.collision_timescale);
        total.maximum_cell_width_over_mean_free_path = std::max(
            total.maximum_cell_width_over_mean_free_path,
            cell.width_over_mean_free_path);
        if (cell.pair_samples != 0) {
            total.minimum_pair_velocity_scale = std::min(
                total.minimum_pair_velocity_scale,
                cell.minimum_pair_velocity_scale);
            total.maximum_pair_velocity_scale = std::max(
                total.maximum_pair_velocity_scale,
                cell.maximum_pair_velocity_scale);
        }
        total.nonlocal_cells += static_cast<std::size_t>(cell.nonlocal);
    }
    if (!std::isfinite(total.minimum_pair_velocity_scale)) {
        // A zero pair scale denotes that no eligible physical-cell pair was
        // sampled.  Positive values are the extrema over the same samples
        // used for the mean-free-path and collision-clock diagnostics.
        total.minimum_pair_velocity_scale = 0.0;
    }

    // The proposal majorant depends on marker mass and internal dispersion,
    // neither of which changes during scattering.  Only capture invalidates a
    // proposal.  Reference mode retains the original mesh-wide clock.  The
    // adaptive mode advances independent cells with independent clocks and
    // batches many local substeps in one GPU launch; a cell stops immediately
    // after capture so no stale majorant is ever reused.
    impl_->cells.resize(static_cast<std::size_t>(cell_count));
    std::vector<DeviceCell> cells(static_cast<std::size_t>(cell_count));
    double local_maximum_frequency = 0.0;
    std::size_t visited = 0;
    bool proposal_dirty = true;

    // Reuse a proposal while all masses and internal dispersions are fixed.
    // Capture marks it dirty and rebuilds both sorted orders and prefixes.
    auto build_proposals = [&]() {
        if (proposal_dirty) {
            impl_->prefix_root.resize(active_count);
            impl_->prefix_mass.resize(active_count);
            impl_->pair_cumulative.resize(active_count);
            build_pair_proposals_kernel<<<cell_blocks, subhalo_threads>>>(
                state.device_view(),
                impl_->active_order.data(),
                impl_->collision_offsets.data(),
                cell_count,
                impl_->mesh_edge.data(),
                impl_->shell_volume.data(),
                device_options,
                table,
                impl_->prefix_root.data(),
                impl_->prefix_mass.data(),
                impl_->pair_cumulative.data(),
                impl_->cells.data());
            detail::check_kernel("GPU subhalo cell majorant failed");
            impl_->cells.download(cells.data(), cells.size());
            local_maximum_frequency = 0.0;
            visited = 0;
            for (const DeviceCell& cell : cells) {
                local_maximum_frequency = std::max(
                    local_maximum_frequency, cell.maximum_trial_frequency);
                if (cell.end - cell.begin >= options.minimum_cell_population) {
                    ++visited;
                }
            }
            proposal_dirty = false;
            ++total.proposal_rebuilds;
        }
    };

    // Add event and energy budgets while retaining worst-pair errors and
    // maximum local work over successive kernel batches.
    auto accumulate = [&](const DeviceSubhaloStats& stats) {
        total.cell_visits += stats.cell_visits;
        total.trial_pairs += stats.trial_pairs;
        total.skipped_already_collided_trials +=
            stats.skipped_already_collided_trials;
        total.accepted_pairs += stats.accepted_pairs;
        total.scattering_events += stats.scattering_events;
        total.capture_events += stats.capture_events;
        total.inner_scattering_events += stats.inner_scattering_events;
        total.inner_capture_events += stats.inner_capture_events;
        total.removed_markers += static_cast<std::size_t>(stats.removed_markers);
        total.represented_scattering_mass += stats.represented_scattering_mass;
        total.represented_capture_mass += stats.represented_capture_mass;
        total.orbital_kinetic_energy_change += stats.orbital_energy_change;
        total.internal_energy_change += stats.internal_energy_change;
        total.maximum_pair_momentum_error = std::max(
            total.maximum_pair_momentum_error,
            bits_to_double(stats.max_pair_momentum_bits));
        total.maximum_pair_total_energy_error = std::max(
            total.maximum_pair_total_energy_error,
            bits_to_double(stats.max_pair_energy_bits));
        total.maximum_cell_substeps = std::max(
            total.maximum_cell_substeps,
            static_cast<std::uint64_t>(stats.maximum_cell_substeps));
    };

    // The reference clock visits every eligible cell on the fastest global
    // frequency, including an MPI maximum when the caller supplies one.
    if (!options.adaptive_cell_timestep) {
        double elapsed = 0.0;
        for (int substep = 0; elapsed < dt; ++substep) {
            if (substep >= options.maximum_substeps) {
                throw std::runtime_error(
                    "GPU subhalo collisions exceeded maximum_substeps.");
            }
            build_proposals();
            double maximum_frequency = local_maximum_frequency;
            if (options.maximum_frequency_reducer != nullptr) {
                maximum_frequency = options.maximum_frequency_reducer(
                    maximum_frequency, options.maximum_frequency_context);
            }
            total.cell_visits += visited;
            const double remaining = dt - elapsed;
            const double subdt = maximum_frequency > 0.0
                ? std::min(
                      remaining, options.collision_cfl / maximum_frequency)
                : remaining;
            if (!(subdt > 0.0) || !std::isfinite(subdt)) {
                throw std::runtime_error(
                    "Invalid GPU subhalo mesh-wide substep.");
            }

            impl_->remove_marker.resize(state.size());
            impl_->remove_marker.zero();
            impl_->collided.resize(active_count);
            impl_->collided.zero();
            impl_->stats.zero();
            collide_subhalo_cells_kernel<<<cell_blocks, subhalo_threads>>>(
                state.device_view(),
                impl_->active_order.data(),
                impl_->cells.data(),
                impl_->prefix_root.data(),
                impl_->prefix_mass.data(),
                impl_->pair_cumulative.data(),
                cell_count,
                device_options,
                table,
                subdt,
                sampler.seed,
                epoch,
                static_cast<std::uint64_t>(substep),
                impl_->collided.data(),
                impl_->remove_marker.data(),
                impl_->stats.data());
            detail::check_kernel(
                "GPU unequal-mass subhalo collision kernel failed");
            DeviceSubhaloStats stats{};
            impl_->stats.download(&stats, 1);
            if (stats.error_flag != 0) {
                throw std::runtime_error(
                    "GPU subhalo collision probability exceeded unity.");
            }
            accumulate(stats);
            ++total.kernel_batches;
            ++total.substeps;
            total.maximum_cell_substeps = total.substeps;
            // Consume the exact remaining interval at the final local step,
            // preventing a tiny arithmetic remainder from generating another step.
            elapsed = subdt >= remaining ? dt : elapsed + subdt;
            // Stable compaction changes every array by the same survivor
            // stencil. Rebuild indices before sampling the changed population.
            if (stats.removed_markers != 0) {
                state.compact_removed_device(
                    impl_->remove_marker.data(),
                    static_cast<std::size_t>(stats.removed_markers));
                build_orders();
                proposal_dirty = true;
            }
        }
    } else {
        // Independent clocks cover the same dt per cell. They remain
        // indexed by physical cell/lane while particle compaction changes only
        // the population and pair proposal within those cells.
        impl_->cell_elapsed.resize(static_cast<std::size_t>(cell_count));
        impl_->cell_substeps.resize(static_cast<std::size_t>(cell_count));
        impl_->cell_elapsed.zero();
        impl_->cell_substeps.zero();

        while (true) {
            build_proposals();
            impl_->remove_marker.resize(state.size());
            impl_->remove_marker.zero();
            impl_->collided_generation.resize(active_count);
            impl_->collided_generation.zero();
            impl_->stats.zero();
            collide_subhalo_cells_adaptive_kernel<<<
                cell_blocks, subhalo_threads>>>(
                state.device_view(),
                impl_->active_order.data(),
                impl_->cells.data(),
                impl_->prefix_root.data(),
                impl_->prefix_mass.data(),
                impl_->pair_cumulative.data(),
                cell_count,
                device_options,
                table,
                dt,
                static_cast<std::uint64_t>(options.maximum_substeps),
                options.adaptive_batch_substeps,
                sampler.seed,
                epoch,
                impl_->cell_elapsed.data(),
                impl_->cell_substeps.data(),
                impl_->collided_generation.data(),
                impl_->remove_marker.data(),
                impl_->stats.data());
            detail::check_kernel(
                "GPU adaptive unequal-mass collision kernel failed");
            DeviceSubhaloStats stats{};
            impl_->stats.download(&stats, 1);
            ++total.kernel_batches;
            if (stats.error_flag == 2) {
                throw std::runtime_error(
                    "GPU adaptive subhalo cell exceeded maximum_substeps.");
            }
            if (stats.error_flag != 0) {
                throw std::runtime_error(
                    "GPU adaptive subhalo collision probability exceeded unity.");
            }
            accumulate(stats);
            total.substeps = total.maximum_cell_substeps;

            // Stable compaction changes every array by the same survivor
            // stencil. Rebuild indices before sampling the changed population.
            if (stats.removed_markers != 0) {
                state.compact_removed_device(
                    impl_->remove_marker.data(),
                    static_cast<std::size_t>(stats.removed_markers));
                build_orders();
                proposal_dirty = true;
            }
            // Completion means every eligible cell has consumed the outer
            // interval, including cells that stopped earlier for a rebuild.
            if (stats.unfinished_cells == 0) {
                break;
            }
        }
    }

    // Verify the installed population after all compactions: capture keeps
    // represented mass, and the internal-energy change equals recorded heat.
    total.active_markers_after = active_count;
    const MarkerDeviceView final_view = state.device_view();
    const double final_mass = device_sum(final_view.mass, final_view.size);
    const double final_internal = device_sum(
        final_view.internal_energy, final_view.size);
    const double mass_scale = std::max(1.0, std::abs(invariant_mass));
    if (std::abs(final_mass - invariant_mass)
        > 512.0 * std::numeric_limits<double>::epsilon() * mass_scale) {
        throw std::runtime_error(
            "GPU subhalo capture failed represented-mass conservation.");
    }
    const double measured_internal_change = final_internal - initial_internal;
    const double internal_scale = std::max(
        1.0,
        std::max(
            std::abs(measured_internal_change),
            std::abs(total.internal_energy_change)));
    if (std::abs(measured_internal_change - total.internal_energy_change)
        > 4096.0 * std::numeric_limits<double>::epsilon() * internal_scale) {
        throw std::runtime_error(
            "GPU subhalo internal-energy ledger failed to close.");
    }
    return total;
}

}  // namespace venus_gpu
