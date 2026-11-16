// Sample elastic material dark matter encounters in spherical radial shells.
// Inputs are device particles, elapsed collision time, a cross-section law,
// collision controls, and a seed/counter. Radius sorting builds local shells;
// disjoint random pairs then sample the Boltzmann rate over local substeps.
// Outputs are updated radial velocity and eta=L^2/2, advanced sampler state,
// and encounter/conservation/locality diagnostics. Masses, radii, IDs, and
// internal energies are fixed by this elastic operator. Each accepted pair
// conserves its center-of-mass momentum and kinetic energy to roundoff.
// Gravity and spatial transport are applied by their separate operator.

#include "venus_gpu/material_sidm.hpp"
#include "venus_gpu/collision_matching.hpp"

#include "cuda_utils.cuh"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
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

constexpr int collision_threads = 256;

// Compact the chosen physical law into scalars usable on both host and
// device. Cross sections are expressed per unit mass throughout this module.
struct DeviceCollisionOptions {
    int model;
    double strength;
    double sigma0;
    double sigma_power;
    double sigma_u0;
    double yukawa_alpha;
    double yukawa_mchi;
    double yukawa_mphi;
};

// Store the radius-ordered index interval and its spherical volume. The
// proposal density compensates for disjoint pair sampling; partner density
// counts the other N-1 particles for the separate mean-free-path diagnostic.
struct DeviceShell {
    std::uint64_t begin;
    std::uint64_t end;
    double inner_edge;
    double outer_edge;
    double volume;
    double number_density;
    double partner_number_density;
    double density;
    double maximum_speed;
    double maximum_mass;
    double mean_free_path;
    double majorant_rate;
    int input_outside_domain;
};

// Accumulate global event counts and conservation residuals; atomic maxima
// retain the largest individual pair error alongside the summed errors.
struct DeviceCollisionStats {
    unsigned long long trial_pairs;
    unsigned long long accepted_pairs;
    double radial_momentum_error;
    double kinetic_energy_error;
    unsigned long long max_pair_momentum_bits;
    unsigned long long max_pair_energy_bits;
    int error_flag;
};

// Only particles currently in the resolved collision population are paired.
struct ActivePredicate {
    const std::uint8_t* state;
    __host__ __device__ bool operator()(std::uint64_t i) const {
        return state[i] == static_cast<std::uint8_t>(MarkerState::active);
    }
};

// Radius supplies spatial locality. Stable identity breaks an exact-radius
// tie deterministically without changing the particle arrays themselves.
struct RadiusIdLess {
    const double* radius;
    const std::uint64_t* id;
    __host__ __device__ bool operator()(
        std::uint64_t first, std::uint64_t second) const {
        if (radius[first] < radius[second]) {
            return true;
        }
        if (radius[second] < radius[first]) {
            return false;
        }
        return id[first] < id[second];
    }
};

// Form target-occupancy shells without splitting an exact-radius tie group.
// A single thread is sufficient: it normally examines one boundary per
// target_shell_markers, and in the degenerate equal-radius case scans the tie
// group only once.  This mirrors the CPU shell builder and prevents zero-volume
// shells for cell-centred marker populations.
__global__ void build_shell_boundaries_kernel(
    const MarkerDeviceView markers,
    const std::uint64_t* order,
    std::uint64_t active_count,
    std::uint64_t target,
    std::uint64_t minimum,
    std::uint64_t* shell_begin,
    std::uint64_t* shell_end,
    std::uint64_t* shell_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    std::uint64_t count = 0;
    for (std::uint64_t begin = 0; begin < active_count;) {
        const std::uint64_t remaining = active_count - begin;
        std::uint64_t end = remaining <= target
            ? active_count : begin + target;
        while (end < active_count
               && markers.r[order[end - 1]] == markers.r[order[end]]) {
            ++end;
        }
        if (active_count - end < minimum) {
            end = active_count;
        }
        shell_begin[count] = begin;
        shell_end[count] = end;
        ++count;
        begin = end;
    }
    *shell_count = count;
}

// Counter keys provide reproducible random words for shell, round, pair,
// particle azimuth, acceptance, and outgoing angle as separate draw purposes.
__host__ __device__ inline std::uint64_t mix64(std::uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

// An open unit interval keeps logarithms or inverse angular laws away from
// singular endpoints. The clamp also protects the rounded upper endpoint.
__device__ inline double uniform_open(
    std::uint64_t seed, std::uint64_t counter) {
    const std::uint64_t word = mix64(seed ^ mix64(counter));
    const std::uint64_t mantissa = word >> 11;
    const double value = (static_cast<double>(mantissa) + 0.5)
        * (1.0 / 9007199254740992.0);
    return fmin(value, 0.99999999999999989);
}

// Evaluate the selected total cross section per unit mass at relative
// speed. Softening or a finite mediator mass regulates the low-speed law.
__host__ __device__ inline double cross_section_per_mass(
    double speed, DeviceCollisionOptions options) {
    if (options.model == static_cast<int>(CrossSectionModel::constant)) {
        return options.strength * options.sigma0;
    }
    if (options.model == static_cast<int>(CrossSectionModel::yukawa_born)) {
        // Integral over the Born differential kernel
        // d sigma/d Omega = S alpha^2 m_chi^2 /
        // [m_phi^2 + (m_chi^2 u^2/2)(1-mu)]^2.
        const double safe_speed = fmax(fabs(speed), 1.0e-14);
        const double mphi2 = options.yukawa_mphi * options.yukawa_mphi;
        const double x = options.yukawa_mchi * options.yukawa_mchi
            * safe_speed * safe_speed / mphi2;
        const double denominator = mphi2 * mphi2 * (1.0 + x);
        return denominator > 0.0
            ? options.strength * 4.0 * pi
                * options.yukawa_alpha * options.yukawa_alpha
                * options.yukawa_mchi * options.yukawa_mchi / denominator
            : 0.0;
    }
    const double safe_speed = fmax(fabs(speed), 1.0e-14);
    const double u2 = safe_speed * safe_speed
        + options.sigma_u0 * options.sigma_u0;
    return options.strength * options.sigma0
        * pow(u2, -0.5 * options.sigma_power);
}

// Bound u*sigma(u) on 0<=u<=u_max. Velocity-dependent laws may peak at
// an interior speed, so compare that analytic extremum with the endpoint
// and add the stated floating-point margin before setting the substep.
__host__ __device__ inline double u_sigma_per_mass_majorant(
    double u_max, DeviceCollisionOptions options) {
    if (!(u_max > 0.0)) {
        return 0.0;
    }
    double candidate = u_max;
    if (options.model == static_cast<int>(CrossSectionModel::power_law)
        && options.sigma_power > 1.0 && options.sigma_u0 > 0.0) {
        candidate = fmin(
            u_max,
            options.sigma_u0 / sqrt(options.sigma_power - 1.0));
    } else if (
        options.model == static_cast<int>(CrossSectionModel::yukawa_born)) {
        candidate = fmin(
            u_max, options.yukawa_mphi / options.yukawa_mchi);
    }
    const double endpoint =
        u_max * cross_section_per_mass(u_max, options);
    const double interior =
        candidate * cross_section_per_mass(candidate, options);
    return fmax(endpoint, interior)
        * (1.0 + 128.0 * 2.2204460492503131e-16);
}

// Invert the normalized angular law. Constant/power-law material models
// use isotropic mu=cos(theta); Born scattering retains its forward preference.
__host__ __device__ inline double sample_scatter_mu(
    double relative_speed,
    double unit,
    DeviceCollisionOptions options) {
    if (options.model != static_cast<int>(CrossSectionModel::yukawa_born)) {
        return fma(2.0, unit, -1.0);
    }
    const double z = fmin(1.0, fmax(0.0, unit));
    if (z <= 0.0) return 1.0;
    if (z >= 1.0) return -1.0;
    // Same angular survival-probability inverse as the CPU material sampler.
    // Factoring the integrated law retains accuracy near isotropic scattering.
    const double speed_ratio =
        fabs(relative_speed) * options.yukawa_mchi / options.yukawa_mphi;
    const double y = speed_ratio * speed_ratio;
    double one_minus_mu;
    if (y <= 1.0) {
        one_minus_mu = 2.0 * z / (1.0 + y * (1.0 - z));
    } else {
        const double inverse_y = 1.0 / y;
        one_minus_mu = 2.0 * z * inverse_y / (inverse_y + (1.0 - z));
    }
    return fmin(1.0, fmax(-1.0, 1.0 - one_minus_mu));
}

// Preserve the largest nonnegative floating-point pair error using an
// integer compare-and-swap, independent of concurrent arrival order.
__device__ inline void atomic_max_bits(
    unsigned long long* address, double value) {
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

// One block measures each shell's total mass, maximum speed, and largest
// particle weight. These give its density and a conservative proposal rate.
__global__ void describe_shells_kernel(
    MarkerDeviceView markers,
    const std::uint64_t* order,
    const std::uint64_t* shell_begin,
    const std::uint64_t* shell_end,
    int shell_count,
    std::uint64_t active_count,
    double domain_inner,
    double domain_outer,
    DeviceCollisionOptions options,
    DeviceShell* shells) {
    const int shell_index = blockIdx.x;
    if (shell_index >= shell_count) {
        return;
    }
    extern __shared__ double shared[];
    double* shared_mass = shared;
    double* shared_speed = shared + blockDim.x;
    double* shared_marker_mass = shared + 2 * blockDim.x;
    const std::uint64_t begin = shell_begin[shell_index];
    const std::uint64_t end = shell_end[shell_index];
    double mass_sum = 0.0;
    double maximum_speed = 0.0;
    double maximum_mass = 0.0;
    for (std::uint64_t local = begin + threadIdx.x;
         local < end;
         local += blockDim.x) {
        const std::uint64_t marker = order[local];
        const double radius = fmax(markers.r[marker], 1.0e-150);
        const double vt = sqrt(2.0 * markers.eta[marker]) / radius;
        mass_sum += markers.mass[marker];
        maximum_speed = fmax(maximum_speed, hypot(markers.w[marker], vt));
        maximum_mass = fmax(maximum_mass, markers.mass[marker]);
    }
    shared_mass[threadIdx.x] = mass_sum;
    shared_speed[threadIdx.x] = maximum_speed;
    shared_marker_mass[threadIdx.x] = maximum_mass;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared_mass[threadIdx.x] += shared_mass[threadIdx.x + offset];
            shared_speed[threadIdx.x] = fmax(
                shared_speed[threadIdx.x], shared_speed[threadIdx.x + offset]);
            shared_marker_mass[threadIdx.x] = fmax(
                shared_marker_mass[threadIdx.x],
                shared_marker_mass[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) {
        return;
    }
    const std::uint64_t count = end - begin;
    const double first_radius = markers.r[order[begin]];
    const double last_radius = markers.r[order[end - 1]];
    // Place neighboring shell faces halfway in r^3, the spherical volume
    // coordinate. The first/last faces use the stated physical domain.
    const double inner = begin == 0
        ? domain_inner
        : cbrt(0.5
               * (pow(markers.r[order[begin - 1]], 3.0)
                  + pow(first_radius, 3.0)));
    const double outer = end == active_count
        ? domain_outer
        : cbrt(0.5
               * (pow(last_radius, 3.0)
                  + pow(markers.r[order[end]], 3.0)));
    DeviceShell shell{};
    shell.begin = begin;
    shell.end = end;
    shell.inner_edge = inner;
    shell.outer_edge = outer;
    shell.volume = (4.0 * pi / 3.0)
        * (outer * outer * outer - inner * inner * inner);
    shell.number_density = static_cast<double>(count) / shell.volume;
    shell.partner_number_density = count > 1
        ? static_cast<double>(count - 1) / shell.volume
        : 0.0;
    shell.density = shared_mass[0] / shell.volume;
    shell.maximum_speed = shared_speed[0];
    shell.maximum_mass = shared_marker_mass[0];
    // Every relative speed is at most twice the shell maximum speed. The
    // largest particle mass bounds the mean mass entering a proposed pair area.
    const double pair_majorant = shell.maximum_mass
        * u_sigma_per_mass_majorant(2.0 * shell.maximum_speed, options);
    // Disjoint matching proposes each physical pair with probability 1/m.
    // For odd count one uniformly chosen marker sits out and m=count;
    // for even count m=count-1. Use this proposal exposure for the CFL,
    // while retaining physical partner density for the mean free path.
    shell.majorant_rate = static_cast<double>(
        collision_matching::round_count(count)) / shell.volume * pair_majorant;
    // The reported locality estimate uses the cross section at reference
    // speed one and physical partner density (N-1)/V. It is a diagnostic
    // convention distinct from the maximum-rate collision probability bound.
    const double mean_mass = shared_mass[0] / static_cast<double>(count);
    const double reference_pair_area = mean_mass
        * cross_section_per_mass(1.0, options);
    shell.mean_free_path = reference_pair_area > 0.0
        ? 1.0 / (shell.partner_number_density * reference_pair_area)
        : static_cast<double>(INFINITY);
    shell.input_outside_domain =
        (begin == 0 && first_radius < domain_inner)
        || (end == active_count && last_radius > domain_outer);
    shells[shell_index] = shell;
}

// One block advances one shell. Each selected factorization round gives
// disjoint pairs, so two threads never update the same particle in that round.
__global__ void collide_shells_kernel(
    MarkerDeviceView markers,
    const std::uint64_t* order,
    const DeviceShell* shells,
    int shell_count,
    DeviceCollisionOptions options,
    const double* shell_subdt,
    double common_subdt,
    std::uint64_t seed,
    std::uint64_t round,
    DeviceCollisionStats* stats) {
    const int shell_index = blockIdx.x;
    if (shell_index >= shell_count) {
        return;
    }
    // Adaptive shells receive their own time increment. A completed shell
    // gets zero while other shells finish the same global collision interval.
    const double subdt = shell_subdt != nullptr
        ? shell_subdt[shell_index] : common_subdt;
    if (!(subdt > 0.0)) {
        return;
    }
    const DeviceShell shell = shells[shell_index];
    const std::uint64_t count = shell.end - shell.begin;
    if (count < 2) {
        return;
    }
    // Share one uniformly selected matching round across the shell block;
    // each thread then obtains its own disjoint pair slot from that round.
    __shared__ std::uint64_t matching_round;
    if (threadIdx.x == 0) {
        const std::uint64_t shell_seed = mix64(
            seed ^ mix64(round)
            ^ mix64(static_cast<std::uint64_t>(shell_index)));
        matching_round = collision_matching::random_round(count, shell_seed);
    }
    __syncthreads();

    const std::uint64_t pairs = count / 2;
    for (std::uint64_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
        // Every unordered pair is represented once over all factorization
        // rounds; pairs within this round have no shared marker. Independent
        // CUDA threads can therefore install both outgoing velocities safely.
        const collision_matching::Pair proposed =
            collision_matching::pair_at(count, matching_round, pair);
        const std::uint64_t first_local = proposed.first;
        const std::uint64_t second_local = proposed.second;
        const std::uint64_t first_index = order[shell.begin + first_local];
        const std::uint64_t second_index = order[shell.begin + second_local];
        const double first_radius = fmax(markers.r[first_index], 1.0e-150);
        const double second_radius = fmax(markers.r[second_index], 1.0e-150);
        const double first_vt =
            sqrt(2.0 * markers.eta[first_index]) / first_radius;
        const double second_vt =
            sqrt(2.0 * markers.eta[second_index]) / second_radius;
        const std::uint64_t event_key = mix64(
            (static_cast<std::uint64_t>(shell_index) << 48)
            ^ (round << 24) ^ pair);
        // Spherical state retains tangential speed but averages over its
        // direction. Independent azimuths reconstruct local 3-D velocities
        // before evaluating the relative speed and collision probability.
        const double first_angle = 2.0 * pi
            * uniform_open(seed ^ UINT64_C(0x123456789abcdef0),
                           event_key ^ markers.id[first_index]);
        const double second_angle = 2.0 * pi
            * uniform_open(seed ^ UINT64_C(0xfedcba9876543210),
                           event_key ^ markers.id[second_index]);
        const double v1x = markers.w[first_index];
        const double v1y = first_vt * cos(first_angle);
        const double v1z = first_vt * sin(first_angle);
        const double v2x = markers.w[second_index];
        const double v2y = second_vt * cos(second_angle);
        const double v2z = second_vt * sin(second_angle);
        const double ux = v1x - v2x;
        const double uy = v1y - v2y;
        const double uz = v1z - v2z;
        const double relative_speed = sqrt(ux * ux + uy * uy + uz * uz);
        const double first_mass = markers.mass[first_index];
        const double second_mass = markers.mass[second_index];
        // Convert cross section per mass into this pair's area. The number
        // of matching rounds divided by shell volume compensates for how
        // often this pair is proposed.
        const double pair_area = 0.5 * (first_mass + second_mass)
            * cross_section_per_mass(relative_speed, options);
        const double proposal_density = static_cast<double>(
            collision_matching::round_count(count)) / shell.volume;
        double probability = proposal_density * relative_speed * pair_area * subdt;
        atomicAdd(&stats->trial_pairs, UINT64_C(1));
        // A probability larger than unity invalidates the selected substep.
        // Only the specified roundoff margin permits clamping at one.
        if (probability > 1.0 + 256.0 * 2.2204460492503131e-16) {
            atomicExch(&stats->error_flag, 1);
            continue;
        }
        probability = fmin(1.0, fmax(0.0, probability));
        if (uniform_open(
                seed ^ UINT64_C(0xa5a5a5a5a5a5a5a5), event_key)
            >= probability) {
            continue;
        }

        // Separate the pair center-of-mass velocity from relative motion.
        // Elastic scattering rotates the relative vector at fixed magnitude.
        const double total_mass = first_mass + second_mass;
        const double px = first_mass * v1x + second_mass * v2x;
        const double py = first_mass * v1y + second_mass * v2y;
        const double pz = first_mass * v1z + second_mass * v2z;
        const double energy_before = 0.5
            * (first_mass * (v1x * v1x + v1y * v1y + v1z * v1z)
               + second_mass * (v2x * v2x + v2y * v2y + v2z * v2z));
        const double center_x = px / total_mass;
        const double center_y = py / total_mass;
        const double center_z = pz / total_mass;
        const double mu = sample_scatter_mu(
            relative_speed,
            uniform_open(
                seed ^ UINT64_C(0x3141592653589793), event_key),
            options);
        const double azimuth = 2.0 * pi
            * uniform_open(
                seed ^ UINT64_C(0x2718281828459045), event_key);
        // Rotate around the incoming relative velocity.  Using a fixed global
        // axis would be valid only for an isotropic kernel; Born scattering is
        // forward-peaked and must preserve the incoming direction as mu->1.
        const double inverse_u = relative_speed > 0.0
            ? 1.0 / relative_speed : 0.0;
        const double hx = relative_speed > 0.0 ? ux * inverse_u : 1.0;
        const double hy = relative_speed > 0.0 ? uy * inverse_u : 0.0;
        const double hz = relative_speed > 0.0 ? uz * inverse_u : 0.0;
        // Choose an axis separated from the incoming direction, then build
        // two perpendicular unit directions for its outgoing polar/azimuthal map.
        const double ax = fabs(hx) < 0.75 ? 1.0 : 0.0;
        const double ay = fabs(hx) < 0.75 ? 0.0 : 1.0;
        const double e1x_raw = hy * 0.0 - hz * ay;
        const double e1y_raw = hz * ax - hx * 0.0;
        const double e1z_raw = hx * ay - hy * ax;
        const double e1_norm = sqrt(
            e1x_raw * e1x_raw + e1y_raw * e1y_raw + e1z_raw * e1z_raw);
        const double e1x = e1x_raw / e1_norm;
        const double e1y = e1y_raw / e1_norm;
        const double e1z = e1z_raw / e1_norm;
        const double e2x = hy * e1z - hz * e1y;
        const double e2y = hz * e1x - hx * e1z;
        const double e2z = hx * e1y - hy * e1x;
        const double sine = sqrt(fmax(0.0, 1.0 - mu * mu));
        const double transverse1 = sine * cos(azimuth);
        const double transverse2 = sine * sin(azimuth);
        const double rx = relative_speed
            * (mu * hx + transverse1 * e1x + transverse2 * e2x);
        const double ry = relative_speed
            * (mu * hy + transverse1 * e1y + transverse2 * e2y);
        const double rz = relative_speed
            * (mu * hz + transverse1 * e1z + transverse2 * e2z);
        // Split the rotated relative velocity inversely with partner mass.
        // This returns both laboratory velocities with the same total momentum.
        const double first_fraction = second_mass / total_mass;
        const double second_fraction = first_mass / total_mass;
        const double out1x = center_x + first_fraction * rx;
        const double out1y = center_y + first_fraction * ry;
        const double out1z = center_z + first_fraction * rz;
        const double out2x = center_x - second_fraction * rx;
        const double out2y = center_y - second_fraction * ry;
        const double out2z = center_z - second_fraction * rz;

        // Store the outgoing radial component and tangential speed through
        // eta=r^2*v_t^2/2 at each unchanged physical particle radius.
        markers.w[first_index] = out1x;
        markers.eta[first_index] = 0.5 * first_radius * first_radius
            * (out1y * out1y + out1z * out1z);
        markers.w[second_index] = out2x;
        markers.eta[second_index] = 0.5 * second_radius * second_radius
            * (out2y * out2y + out2z * out2z);
        atomicAdd(&stats->accepted_pairs, UINT64_C(1));

        // Recompute invariants from the installed outgoing values, retaining
        // total and worst-pair residuals for numerical conservation checks.
        const double after_px = first_mass * out1x + second_mass * out2x;
        const double after_py = first_mass * out1y + second_mass * out2y;
        const double after_pz = first_mass * out1z + second_mass * out2z;
        const double energy_after = 0.5
            * (first_mass
                   * (out1x * out1x + out1y * out1y + out1z * out1z)
               + second_mass
                   * (out2x * out2x + out2y * out2y + out2z * out2z));
        const double radial_error = after_px - px;
        const double energy_error = energy_after - energy_before;
        atomicAdd(&stats->radial_momentum_error, radial_error);
        atomicAdd(&stats->kinetic_energy_error, energy_error);
        const double momentum_error = sqrt(
            (after_px - px) * (after_px - px)
            + (after_py - py) * (after_py - py)
            + (after_pz - pz) * (after_pz - pz));
        atomic_max_bits(&stats->max_pair_momentum_bits, momentum_error);
        atomic_max_bits(&stats->max_pair_energy_bits, fabs(energy_error));
    }
}

// Check the selected law's parameter domain and finite-work/CFL controls
// before any state-changing kernel can run.
DeviceCollisionOptions make_device_options(const CollisionOptions& options) {
    const bool valid_model = options.cross_section == CrossSectionModel::constant
        || options.cross_section == CrossSectionModel::power_law
        || options.cross_section == CrossSectionModel::yukawa_born;
    const bool valid_cross_section =
        (options.cross_section == CrossSectionModel::constant
            && options.sigma0 >= 0.0 && std::isfinite(options.sigma0))
        || (options.cross_section == CrossSectionModel::power_law
            && options.sigma0 >= 0.0 && std::isfinite(options.sigma0)
            && options.sigma_power >= 0.0
            && std::isfinite(options.sigma_power)
            && options.sigma_u0 >= 0.0
            && std::isfinite(options.sigma_u0))
        || (options.cross_section == CrossSectionModel::yukawa_born
            && options.yukawa_alpha >= 0.0
            && std::isfinite(options.yukawa_alpha)
            && options.yukawa_mchi > 0.0
            && std::isfinite(options.yukawa_mchi)
            && options.yukawa_mphi > 0.0
            && std::isfinite(options.yukawa_mphi));
    if (!valid_model || !valid_cross_section
        || !(options.collision_strength >= 0.0)
        || !std::isfinite(options.collision_strength)
        || options.target_shell_markers < 2
        || options.minimum_shell_markers < 2
        || options.minimum_shell_markers > options.target_shell_markers
        || !(options.collision_cfl > 0.0 && options.collision_cfl <= 1.0)
        || options.maximum_substeps < 1) {
        throw std::runtime_error("Invalid GPU weighted-PIC collision options.");
    }
    DeviceCollisionOptions result;
    result.model = static_cast<int>(options.cross_section);
    result.strength = options.collision_strength;
    result.sigma0 = options.sigma0;
    result.sigma_power = options.sigma_power;
    result.sigma_u0 = options.sigma_u0;
    result.yukawa_alpha = options.yukawa_alpha;
    result.yukawa_mchi = options.yukawa_mchi;
    result.yukawa_mphi = options.yukawa_mphi;
    return result;
}

}  // namespace

using detail::DeviceArray;

// Keep sorted-index, shell-geometry, local-clock, and statistics workspaces
// across calls. The physical particle arrays remain owned by WpicState.
struct CollisionModule::Impl {
    explicit Impl(const RadialMesh& radial_mesh) : mesh(radial_mesh), stats(1) {}

    RadialMesh mesh;
    DeviceArray<std::uint64_t> active_order;
    DeviceArray<std::uint64_t> shell_begin;
    DeviceArray<std::uint64_t> shell_end;
    DeviceArray<std::uint64_t> shell_count;
    DeviceArray<DeviceShell> shells;
    DeviceArray<double> shell_subdt;
    DeviceArray<DeviceCollisionStats> stats;
};

CollisionModule::CollisionModule(const RadialMesh& mesh)
    : impl_(new Impl(mesh)) {}

CollisionModule::~CollisionModule() = default;
CollisionModule::CollisionModule(CollisionModule&&) noexcept = default;
CollisionModule& CollisionModule::operator=(CollisionModule&&) noexcept = default;

// Cover the supplied collision interval at fixed radii. First build the
// spatial population and rate bounds, then sample local or common substeps.
CollisionDiagnostics CollisionModule::apply(
    WpicState& state,
    double dt,
    const CollisionOptions& options,
    CollisionSamplerState& sampler) {
    if (!(dt >= 0.0) || !std::isfinite(dt)) {
        throw std::runtime_error("Invalid GPU collision interval.");
    }
    const DeviceCollisionOptions device_options = make_device_options(options);
    CollisionDiagnostics diagnostics;
    // A zero physical rate or zero interval consumes no stochastic work.
    // This also preserves the sampler on a collisionless control call.
    const bool zero_cross_section =
        options.cross_section == CrossSectionModel::yukawa_born
        ? options.yukawa_alpha == 0.0
        : options.sigma0 == 0.0;
    if (dt == 0.0 || zero_cross_section
        || options.collision_strength == 0.0 || state.size() < 2) {
        return diagnostics;
    }

    const MarkerDeviceView markers = state.device_view();
    impl_->active_order.resize(markers.size);
    thrust::counting_iterator<std::uint64_t> first(UINT64_C(0));
    thrust::device_ptr<std::uint64_t> output(impl_->active_order.data());
    const thrust::device_ptr<std::uint64_t> active_end = thrust::copy_if(
        first,
        first + static_cast<std::uint64_t>(markers.size),
        output,
        ActivePredicate{markers.state});
    const std::size_t active_count =
        static_cast<std::size_t>(active_end - output);
    diagnostics.active_markers = active_count;
    if (active_count < 2) {
        return diagnostics;
    }
    // Sort indices rather than records, retaining the stored particle order
    // and the ID correspondence used by checkpoints and other modules.
    thrust::sort(
        output,
        output + active_count,
        RadiusIdLess{markers.r, markers.id});

    const std::size_t maximum_shells = 1
        + (active_count - 1) / options.target_shell_markers;
    impl_->shell_begin.resize(maximum_shells);
    impl_->shell_end.resize(maximum_shells);
    impl_->shell_count.resize(1);
    build_shell_boundaries_kernel<<<1, 1>>>(
        markers,
        impl_->active_order.data(),
        static_cast<std::uint64_t>(active_count),
        static_cast<std::uint64_t>(options.target_shell_markers),
        static_cast<std::uint64_t>(options.minimum_shell_markers),
        impl_->shell_begin.data(),
        impl_->shell_end.data(),
        impl_->shell_count.data());
    detail::check_kernel("GPU collision shell-boundary kernel failed");
    std::uint64_t shell_count_u64 = 0;
    impl_->shell_count.download(&shell_count_u64, 1);
    if (shell_count_u64 == 0
        || shell_count_u64 > static_cast<std::uint64_t>(INT_MAX)) {
        throw std::runtime_error("GPU collision shell count exceeds INT_MAX.");
    }
    const int shell_count = static_cast<int>(shell_count_u64);
    diagnostics.shells = static_cast<std::size_t>(shell_count_u64);
    impl_->shells.resize(diagnostics.shells);
    describe_shells_kernel<<<
        shell_count,
        collision_threads,
        3 * collision_threads * sizeof(double)>>>(
        markers, impl_->active_order.data(), impl_->shell_begin.data(),
        impl_->shell_end.data(), shell_count,
        static_cast<std::uint64_t>(active_count), impl_->mesh.edge.front(),
        impl_->mesh.edge.back(), device_options, impl_->shells.data());
    detail::check_kernel("GPU collision shell-description kernel failed");
    std::vector<DeviceShell> host_shells(diagnostics.shells);
    impl_->shells.download(host_shells.data(), host_shells.size());

    // Validate shell geometry and measure width/mean-free-path before
    // applying any collision. The optional locality stop is a spatial check.
    double maximum_rate = 0.0;
    for (std::size_t i = 0; i < host_shells.size(); ++i) {
        if (host_shells[i].input_outside_domain != 0
            || !(host_shells[i].outer_edge > host_shells[i].inner_edge)
            || !(host_shells[i].volume > 0.0)
            || !std::isfinite(host_shells[i].volume)
            || !(host_shells[i].majorant_rate >= 0.0)
            || !std::isfinite(host_shells[i].majorant_rate)) {
            throw std::runtime_error(
                "GPU material-collision markers must lie inside the radial "
                "mesh and form finite, positive-volume shells.");
        }
        maximum_rate = std::max(maximum_rate, host_shells[i].majorant_rate);
        const double ratio = std::isfinite(host_shells[i].mean_free_path)
            && host_shells[i].mean_free_path > 0.0
            ? (host_shells[i].outer_edge - host_shells[i].inner_edge)
                / host_shells[i].mean_free_path
            : 0.0;
        diagnostics.maximum_shell_width_over_mean_free_path = std::max(
            diagnostics.maximum_shell_width_over_mean_free_path, ratio);
        if (ratio > 1.0) {
            ++diagnostics.nonlocal_shells;
        }
    }
    if (options.fail_on_nonlocal_shell && diagnostics.nonlocal_shells != 0) {
        throw std::runtime_error(
            "A GPU collision shell exceeds its estimated mean free path. "
            "Increase marker count or lower target_shell_markers.");
    }
    impl_->stats.zero();
    std::uint64_t substeps = 0;
    if (options.adaptive_shell_timestep) {
        // Elastic scattering freezes marker radii and masses, hence the
        // sorted shell layout is reusable for the complete operator batch.
        // Only velocities and their safe u*Sigma majorants are refreshed.
        std::vector<double> elapsed(diagnostics.shells, 0.0);
        std::vector<double> local_subdt(diagnostics.shells, 0.0);
        std::vector<std::uint64_t> local_substeps(diagnostics.shells, 0);
        impl_->shell_subdt.resize(diagnostics.shells);
        bool incomplete = true;
        std::uint64_t wave = 0;
        while (incomplete) {
            if (wave != 0) {
                describe_shells_kernel<<<
                    shell_count,
                    collision_threads,
                    3 * collision_threads * sizeof(double)>>>(
                    markers, impl_->active_order.data(),
                    impl_->shell_begin.data(), impl_->shell_end.data(),
                    shell_count, static_cast<std::uint64_t>(active_count),
                    impl_->mesh.edge.front(), impl_->mesh.edge.back(),
                    device_options, impl_->shells.data());
                detail::check_kernel(
                    "GPU adaptive collision shell refresh failed");
                impl_->shells.download(host_shells.data(), host_shells.size());
            }
            incomplete = false;
            bool active_wave = false;
            // Each shell advances by min(remaining time, CFL/rate). Its
            // own substep count limits work even when another shell finishes.
            for (std::size_t shell = 0; shell < diagnostics.shells; ++shell) {
                local_subdt[shell] = 0.0;
                const double remaining = dt - elapsed[shell];
                if (!(remaining > 0.0)) {
                    continue;
                }
                if (local_substeps[shell]
                    >= static_cast<std::uint64_t>(options.maximum_substeps)) {
                    throw std::runtime_error(
                        "GPU adaptive collision shell exceeded its local "
                        "maximum_substeps.");
                }
                const double rate = host_shells[shell].majorant_rate;
                if (!(rate >= 0.0) || !std::isfinite(rate)) {
                    throw std::runtime_error(
                        "GPU adaptive collision shell has invalid rate.");
                }
                local_subdt[shell] = rate > 0.0
                    ? std::min(remaining, options.collision_cfl / rate)
                    : remaining;
                if (!(local_subdt[shell] > 0.0)
                    || !std::isfinite(local_subdt[shell])) {
                    throw std::runtime_error(
                        "GPU adaptive collision shell has invalid timestep.");
                }
                active_wave = true;
            }
            if (!active_wave) {
                break;
            }
            if (wave >= UINT64_MAX - sampler.next_counter) {
                throw std::runtime_error(
                    "GPU collision sampler counter exhausted.");
            }
            // One wave can contain different positive shell increments.
            // The common wave key advances once, while each shell key keeps
            // its stochastic stream distinct within that wave.
            impl_->shell_subdt.upload(
                local_subdt.data(), local_subdt.size());
            collide_shells_kernel<<<shell_count, collision_threads>>>(
                markers, impl_->active_order.data(), impl_->shells.data(),
                shell_count, device_options, impl_->shell_subdt.data(), 0.0,
                sampler.seed, sampler.next_counter + wave,
                impl_->stats.data());
            detail::check_kernel("GPU adaptive collision pairing kernel failed");
            for (std::size_t shell = 0; shell < diagnostics.shells; ++shell) {
                if (!(local_subdt[shell] > 0.0)) {
                    continue;
                }
                const double remaining = dt - elapsed[shell];
                // Land exactly on dt when consuming the remainder, avoiding
                // a final unrepresentable residue in the local clock.
                elapsed[shell] = local_subdt[shell] >= remaining
                    ? dt : elapsed[shell] + local_subdt[shell];
                ++local_substeps[shell];
                incomplete = incomplete || elapsed[shell] < dt;
            }
            ++wave;
        }
        substeps = wave;
        for (const std::uint64_t count : local_substeps) {
            diagnostics.shell_visits += count;
        }
    } else {
        // The reference branch divides the whole interval into common
        // substeps using the largest initial shell rate. The kernel still
        // rejects any realized pair probability exceeding its allowed bound.
        const double required_substeps = maximum_rate > 0.0
            ? std::ceil(dt * maximum_rate / options.collision_cfl) : 1.0;
        if (!std::isfinite(required_substeps)
            || required_substeps > options.maximum_substeps) {
            throw std::runtime_error(
                "GPU collision interval exceeded maximum_substeps.");
        }
        substeps = static_cast<std::uint64_t>(
            std::max(1, static_cast<int>(required_substeps)));
        if (substeps > UINT64_MAX - sampler.next_counter) {
            throw std::runtime_error("GPU collision sampler counter exhausted.");
        }
        const double subdt = dt / static_cast<double>(substeps);
        for (std::uint64_t round = 0; round < substeps; ++round) {
            collide_shells_kernel<<<shell_count, collision_threads>>>(
                markers, impl_->active_order.data(), impl_->shells.data(),
                shell_count, device_options, nullptr, subdt, sampler.seed,
                sampler.next_counter + round, impl_->stats.data());
            detail::check_kernel("GPU collision pairing kernel failed");
        }
        diagnostics.shell_visits = substeps * diagnostics.shells;
    }
    // This download closes all collision kernels before exposing counts or
    // advancing the stored sampler counter to the next unused round.
    DeviceCollisionStats stats{};
    impl_->stats.download(&stats, 1);
    if (stats.error_flag != 0) {
        throw std::runtime_error(
            "GPU collision probability exceeded unity despite subcycling.");
    }
    diagnostics.substeps = substeps;
    diagnostics.trial_pairs = stats.trial_pairs;
    diagnostics.accepted_pairs = stats.accepted_pairs;
    diagnostics.radial_momentum_error = stats.radial_momentum_error;
    diagnostics.kinetic_energy_error = stats.kinetic_energy_error;
    std::uint64_t momentum_bits = stats.max_pair_momentum_bits;
    std::uint64_t energy_bits = stats.max_pair_energy_bits;
    std::memcpy(
        &diagnostics.max_pair_momentum_error,
        &momentum_bits,
        sizeof(double));
    std::memcpy(
        &diagnostics.max_pair_kinetic_energy_error,
        &energy_bits,
        sizeof(double));
    sampler.next_counter += substeps;
    return diagnostics;
}

// Public host evaluation uses the same checked law as the CUDA rate path.
double material_cross_section_per_mass(
    double relative_speed,
    const CollisionOptions& options) {
    if (!(relative_speed >= 0.0) || !std::isfinite(relative_speed)) {
        throw std::runtime_error("Invalid material relative speed.");
    }
    return cross_section_per_mass(relative_speed, make_device_options(options));
}

// Public host evaluation exposes the same inverse angular CDF for
// comparisons with reference calculations and distribution checks.
double material_scatter_cosine(
    double relative_speed,
    double unit,
    const CollisionOptions& options) {
    if (!(relative_speed >= 0.0) || !std::isfinite(relative_speed)
        || !std::isfinite(unit)
        || unit < 0.0 || unit > 1.0) {
        throw std::runtime_error("Invalid material scatter-CDF input.");
    }
    return sample_scatter_mu(
        relative_speed, unit, make_device_options(options));
}

}  // namespace venus_gpu
