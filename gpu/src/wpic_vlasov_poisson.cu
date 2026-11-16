// Solve spherical Vlasov-Poisson evolution for weighted particles on one GPU.
// Inputs are particle radii/velocities/angular momenta, a radial mesh, gravity,
// expansion, and boundary options. The module deposits shell moments, solves
// for enclosed mass and potential, and advances collisionless characteristics.
// Outputs are updated device particles plus an explicitly downloadable field
// and mass ledger. Persistent work arrays keep the field calculation on the
// device. Particle mass, ID, and eta=L^2/2 remain fixed along these orbits.
// MPI performs its global moment reduction before using the common solve path.

#include "venus_gpu/vlasov_poisson.hpp"
#include "ev/central_drift.hpp"

#include "cuda_utils.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace venus_gpu {
namespace {

constexpr int threads_per_block = 256;

// Pass only the scalar physical choices needed by kernels. Background
// density enters the Poisson source only when explicit subtraction is selected.
struct DeviceVPOptions {
    double gravity;
    double central_mass;
    double rho_bar;
    int particle_shape;
    int inner_boundary;
    int outer_boundary;
    int subtract_background;
    int carry_inside;
};

// Locate the first stored coordinate strictly greater than x. This common
// binary search assigns a radius or volume coordinate to its adjacent faces.
__device__ inline int upper_cell(const double* sorted, int count, double x) {
    int first = 0;
    int last = count;
    while (first < last) {
        const int middle = first + (last - first) / 2;
        if (x < sorted[middle]) {
            last = middle;
        } else {
            first = middle + 1;
        }
    }
    return first;
}

// One particle cloud contributes the same assigned mass to every moment.
// The squared assigned weight records the effective number in the shell;
// atomic sums allow several particles to contribute to a shared cell.
__device__ inline void add_moments(
    int cell,
    double weight,
    double marker_mass,
    double radial_velocity,
    double tangential_speed_squared,
    double* shell_mass,
    double* shell_momentum,
    double* shell_radial_second,
    double* shell_tangential_second,
    double* shell_energy_flux,
    double* shell_mass_square,
    double* ledger) {
    if (!(weight > 0.0)) {
        return;
    }
    const double contribution = marker_mass * weight;
    const double w2 = radial_velocity * radial_velocity;
    atomicAdd(shell_mass + cell, contribution);
    atomicAdd(shell_momentum + cell, contribution * radial_velocity);
    atomicAdd(shell_radial_second + cell, contribution * w2);
    atomicAdd(
        shell_tangential_second + cell,
        contribution * tangential_speed_squared);
    atomicAdd(
        shell_energy_flux + cell,
        0.5 * contribution * radial_velocity
            * (w2 + tangential_speed_squared));
    atomicAdd(shell_mass_square + cell, contribution * contribution);
    atomicAdd(ledger + 1, contribution);  // deposited mesh mass
}

// One CUDA thread handles one marker.  Assignment is conservative in the
// volume coordinate q=r^3/3: CIC partitions between adjacent centers, whereas
// TSC renormalizes its three-cell stencil by shell volume at a boundary.  The
// four-entry ledger separately tracks resolved, deposited, hidden-inner, and
// escaped mass so the host can check closure after the asynchronous kernels.
__global__ void deposit_kernel(
    MarkerDeviceView markers,
    const double* edge,
    const double* q_center,
    const double* shell_volume,
    int cells,
    DeviceVPOptions options,
    double* shell_mass,
    double* shell_momentum,
    double* shell_radial_second,
    double* shell_tangential_second,
    double* shell_energy_flux,
    double* shell_mass_square,
    double* ledger,
    int* error_flag) {
    const std::size_t marker_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (marker_index >= markers.size) {
        return;
    }
    const double mass = markers.mass[marker_index];
    // Boundary carriers contribute to separate extensive mass ledgers. Only
    // active resolved particles enter the shell velocity-moment deposition.
    const std::uint8_t state = markers.state[marker_index];
    if (state == static_cast<std::uint8_t>(MarkerState::delayed_inner)) {
        atomicAdd(ledger + 2, mass);
        return;
    }
    if (state == static_cast<std::uint8_t>(MarkerState::escaped)) {
        atomicAdd(ledger + 3, mass);
        return;
    }

    double radius = markers.r[marker_index];
    if (!isfinite(radius) || radius < 0.0 || radius > edge[cells]) {
        atomicExch(error_flag, 1);
        return;
    }
    const double radial_velocity = markers.w[marker_index];
    const double radius_squared = fmax(radius * radius, 1.0e-300);
    const double tangential_speed_squared =
        2.0 * markers.eta[marker_index] / radius_squared;

    // Split a cloud crossing the inner face smoothly into resolved and
    // unresolved source mass. Evaluate its tangential speed at its true radius
    // before moving the deposition coordinate onto the first mesh face.
    double resolved_fraction = 1.0;
    if (options.carry_inside != 0 && radius < edge[0]) {
        const double face_q = edge[0] * edge[0] * edge[0] / 3.0;
        const double first_width = 2.0 * (q_center[0] - face_q);
        const double transition_width = fmin(first_width, face_q);
        if (!(transition_width > 0.0)) {
            atomicExch(error_flag, 2);
            return;
        }
        const double q = radius * radius * radius / 3.0;
        const double s = fmin(
            1.0,
            fmax(0.0, (q - (face_q - transition_width)) / transition_width));
        resolved_fraction = s * s * (3.0 - 2.0 * s);
        atomicAdd(ledger + 2, mass * (1.0 - resolved_fraction));
        radius = edge[0];
    } else if (radius < edge[0]) {
        atomicExch(error_flag, 3);
        return;
    }
    atomicAdd(ledger, mass * resolved_fraction);
    if (!(resolved_fraction > 0.0)) {
        return;
    }

    const double q = radius * radius * radius / 3.0;
    // CIC divides the resolved cloud mass between adjacent q centers. At a
    // mesh end its complete resolved share belongs to the nearest cell.
    if (options.particle_shape == static_cast<int>(ParticleShape::cic)) {
        if (cells == 1 || q <= q_center[0]) {
            add_moments(
                0, resolved_fraction, mass, radial_velocity,
                tangential_speed_squared, shell_mass, shell_momentum,
                shell_radial_second, shell_tangential_second,
                shell_energy_flux, shell_mass_square, ledger);
            return;
        }
        if (q >= q_center[cells - 1]) {
            add_moments(
                cells - 1, resolved_fraction, mass, radial_velocity,
                tangential_speed_squared, shell_mass, shell_momentum,
                shell_radial_second, shell_tangential_second,
                shell_energy_flux, shell_mass_square, ledger);
            return;
        }
        const int right = upper_cell(q_center, cells, q);
        const int left = right - 1;
        const double fraction =
            (q - q_center[left]) / (q_center[right] - q_center[left]);
        add_moments(
            left, resolved_fraction * (1.0 - fraction), mass,
            radial_velocity, tangential_speed_squared, shell_mass,
            shell_momentum, shell_radial_second, shell_tangential_second,
            shell_energy_flux, shell_mass_square, ledger);
        add_moments(
            right, resolved_fraction * fraction, mass, radial_velocity,
            tangential_speed_squared, shell_mass, shell_momentum,
            shell_radial_second, shell_tangential_second, shell_energy_flux,
            shell_mass_square, ledger);
        return;
    }

    if (cells == 1) {
        add_moments(
            0, resolved_fraction, mass, radial_velocity,
            tangential_speed_squared, shell_mass, shell_momentum,
            shell_radial_second, shell_tangential_second, shell_energy_flux,
            shell_mass_square, ledger);
        return;
    }
    // Express q on the nonuniform center grid as a continuous logical cell
    // coordinate, then build the three-point quadratic TSC cloud there.
    double logical = 0.0;
    if (q <= q_center[0]) {
        logical = (q - q_center[0]) / (q_center[1] - q_center[0]);
    } else if (q >= q_center[cells - 1]) {
        logical = static_cast<double>(cells - 1)
            + (q - q_center[cells - 1])
                / (q_center[cells - 1] - q_center[cells - 2]);
    } else {
        const int right = upper_cell(q_center, cells, q);
        const int left = right - 1;
        logical = static_cast<double>(left)
            + (q - q_center[left]) / (q_center[right] - q_center[left]);
    }
    // The three nonnegative shape values describe one cloud. Multiply by
    // available shell volumes and normalize to conserve its resolved mass.
    const int nearest = static_cast<int>(floor(logical + 0.5));
    const double displacement = logical - static_cast<double>(nearest);
    const double raw[3] = {
        0.5 * (0.5 - displacement) * (0.5 - displacement),
        0.75 - displacement * displacement,
        0.5 * (0.5 + displacement) * (0.5 + displacement),
    };
    // The sum uses only cells inside the mesh, preserving cloud mass when
    // a boundary removes one of the nominal three shape contributions.
    double normalization = 0.0;
    for (int offset = -1; offset <= 1; ++offset) {
        const int cell = nearest + offset;
        if (cell >= 0 && cell < cells) {
            normalization += raw[offset + 1] * shell_volume[cell];
        }
    }
    if (!(normalization > 0.0)) {
        atomicExch(error_flag, 4);
        return;
    }
    for (int offset = -1; offset <= 1; ++offset) {
        const int cell = nearest + offset;
        if (cell >= 0 && cell < cells) {
            const double weight = resolved_fraction
                * raw[offset + 1] * shell_volume[cell] / normalization;
            add_moments(
                cell, weight, mass, radial_velocity,
                tangential_speed_squared, shell_mass, shell_momentum,
                shell_radial_second, shell_tangential_second,
                shell_energy_flux, shell_mass_square, ledger);
        }
    }
}

// Treat each finite-volume source as constant line mass dM/dr.  Prefix sums of
// source_shell then give enclosed mass on faces, and the following two kernels
// integrate GM(r)/r^2 analytically within each cell to obtain force and a
// continuous potential without a cell-center finite-difference derivative.
__global__ void source_shell_kernel(
    const double* deposited_mass,
    const double* edge,
    const double* radius_center,
    int cells,
    DeviceVPOptions options,
    double* source_shell,
    double* line_density) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells) {
        return;
    }
    const double width = edge[i + 1] - edge[i];
    double mass = deposited_mass[i];
    // This branch removes the homogeneous density represented by the input
    // weights, using the selected radial source discretization once.
    if (options.subtract_background != 0) {
        mass -= 4.0 * pi * radius_center[i] * radius_center[i]
            * width * options.rho_bar;
    }
    source_shell[i] = mass;
    line_density[i] = mass / width;
}

// Add the point source and hidden inner mass to every enclosed-mass face.
// The first face then carries the complete unresolved interior source.
__global__ void add_central_mass_kernel(
    double* source_mass_edge,
    int cells,
    double central_source,
    const double* ledger) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > cells) {
        return;
    }
    if (i == 0) {
        source_mass_edge[0] = central_source + ledger[2];
    } else {
        source_mass_edge[i] += central_source + ledger[2];
    }
}

__global__ void field_and_integral_kernel(
    const double* edge,
    const double* radius_center,
    const double* source_mass_edge,
    const double* line_density,
    int cells,
    double gravity,
    double* phi_r,
    double* full_cell_force_integral) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells) {
        return;
    }
    const double left = edge[i];
    const double right = edge[i + 1];
    const double center = radius_center[i];
    const double lambda = line_density[i];
    const double mass_center = source_mass_edge[i]
        + lambda * (center - left);
    phi_r[i] = gravity * mass_center / (center * center);
    // Within this cell M(r)=constant+lambda*r. Integrating G*M(r)/r^2
    // analytically gives the potential difference across its two faces.
    const double constant = source_mass_edge[i] - lambda * left;
    full_cell_force_integral[i] = gravity
        * (constant * (1.0 / left - 1.0 / right)
           + lambda * log(right / left));
}

// Start with the exterior vacuum value -G*M_outer/r_outer, which fixes
// Phi(infinity)=0. Summed inward force integrals give face and center values.
__global__ void potential_kernel(
    const double* edge,
    const double* radius_center,
    const double* source_mass_edge,
    const double* line_density,
    const double* force_integral_suffix,
    int cells,
    double gravity,
    double* phi_edge,
    double* phi) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const double outer_phi =
        -gravity * source_mass_edge[cells] / edge[cells];
    if (i == cells) {
        phi_edge[cells] = outer_phi;
        return;
    }
    if (i > cells) {
        return;
    }
    phi_edge[i] = outer_phi - force_integral_suffix[i];
    const double left = edge[i];
    const double right = edge[i + 1];
    const double center = radius_center[i];
    const double lambda = line_density[i];
    const double constant = source_mass_edge[i] - lambda * left;
    const double center_to_right = gravity
        * (constant * (1.0 / center - 1.0 / right)
           + lambda * log(right / center));
    const double right_phi = (i + 1 == cells)
        ? outer_phi
        : outer_phi - force_integral_suffix[i + 1];
    phi[i] = right_phi - center_to_right;
}

// Neff measures the number of equal weights carrying the same shell mass:
// (sum assigned mass)^2/sum(assigned mass^2). Empty shells have zero count.
__global__ void derived_moment_kernel(
    const double* shell_mass,
    const double* shell_mass_square,
    int cells,
    double* effective_number) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cells) {
        effective_number[i] = shell_mass_square[i] > 0.0
            ? shell_mass[i] * shell_mass[i] / shell_mass_square[i]
            : 0.0;
    }
}

// Return dPhi/dr at the particle radius from the same constant-line-mass
// reconstruction used by the mesh field, with explicit interior/exterior laws.
__device__ inline double sample_force(
    double radius,
    const double* edge,
    const double* radius_center,
    const double* source_mass_edge,
    const double* line_density,
    const double* phi_r,
    int cells,
    DeviceVPOptions options) {
    radius = fabs(radius);
    if (radius < edge[0]) {
        // The unresolved uniform core is normalized by mass at the inner
        // face. Exterior shell mass exerts no force inside an empty cavity.
        const double inner = edge[0];
        const double core_coefficient = options.gravity
            * (source_mass_edge[0] - options.central_mass)
            / (inner * inner * inner);
        const double safe_radius = fmax(radius, 1.0e-150);
        return options.central_mass > 0.0
            ? options.gravity * options.central_mass
                    / (safe_radius * safe_radius)
                + core_coefficient * radius
            : core_coefficient * radius;
    }
    // Beyond the outer face the stored total source produces a vacuum
    // inverse-square field; inside a cell interpolate enclosed mass linearly.
    if (radius > edge[cells]) {
        return options.gravity * source_mass_edge[cells]
            / (radius * radius);
    }
    int cell = upper_cell(edge, cells + 1, radius) - 1;
    cell = max(0, min(cells - 1, cell));
    const double mass = source_mass_edge[cell]
        + line_density[cell] * (radius - edge[cell]);
    return options.gravity * mass / (radius * radius);
}

// Apply the first gravitational half kick in canonical expanding momentum,
// then drift the central-force characteristic exactly.  The drift coefficient
// is the analytic integral of a(t)^-2; reconstructing radial/tangential axes
// through hypot preserves eta while avoiding a coordinate singularity at a
// turning point. Potential wall crossings use the shared CPU/device solver
// to reflect at the actual impact time and drift through the remaining time.
__global__ void begin_kdk_kernel(
    MarkerDeviceView markers,
    const double* edge,
    const double* radius_center,
    const double* source_mass_edge,
    const double* line_density,
    const double* phi_r,
    int cells,
    DeviceVPOptions options,
    double dt,
    double start_scale,
    double end_scale,
    double drift_tau,
    double expansion_alpha,
    double drift,
    int* error_flag) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= markers.size
        || markers.state[i] != static_cast<std::uint8_t>(MarkerState::active)) {
        return;
    }
    const bool radial_origin = options.inner_boundary == 1 && markers.eta[i] == 0.0;
    if (!(markers.r[i] >= 0.0)
        || (markers.r[i] == 0.0 && (!radial_origin || options.central_mass != 0.0))) {
        atomicExch(error_flag, 10);
        return;
    }
    const double force = sample_force(
        markers.r[i], edge, radius_center, source_mass_edge, line_density,
        phi_r, cells, options);
    // Comoving radius x=r/a and canonical radial momentum p=a*w remove
    // the explicit Hubble drift/drag from the free characteristic substep.
    const double x = markers.r[i] / start_scale;
    // L=sqrt(2*eta) is constant. The free central drift is a straight path
    // in its orbital plane, whose radial and tangential projections follow.
    const double angular_momentum = sqrt(2.0 * markers.eta[i]);
    double momentum = start_scale * markers.w[i]
        - 0.5 * dt * start_scale * force;
    const double radial_axis = x + drift * momentum;
    const double tangential_axis = radial_origin ? 0.0 : drift * angular_momentum / x;
    const double new_x = hypot(radial_axis, tangential_axis);
    if (!isfinite(new_x)
        || (!(new_x > 0.0) && options.inner_boundary == 1 && !radial_origin)
        || (radial_origin && options.central_mass != 0.0 && radial_axis <= 0.0)) {
        atomicExch(error_flag, 11);
        return;
    }
    const double initial_momentum = momentum;
    // Pure radial motion admits a signed continuation through r=0. A finite
    // angular momentum instead turns at its centrifugal barrier.
    if (radial_origin) {
        // The signed radial trajectory continues through the regular origin.
        // A step ending exactly there stores the outgoing radial velocity.
        if (radial_axis < 0.0) momentum = -momentum;
        else if (radial_axis == 0.0 && dt > 0.0) momentum = fabs(momentum);
    } else {
        momentum = new_x > 0.0 ? (momentum * radial_axis
                    + angular_momentum / x * tangential_axis) / new_x : 0.0;
    }
    double radius = end_scale * new_x;
    double velocity = momentum / end_scale;
    // The comoving free trajectory gives a conservative radial interval.
    // Preserve the existing arithmetic when that interval excludes both walls.
    const double tangential_momentum = radial_origin ? 0.0 : angular_momentum / x;
    const double speed2 = initial_momentum * initial_momentum
        + tangential_momentum * tangential_momentum;
    const double nearest_drift = speed2 > 0.0
        ? fmax(0.0, fmin(drift, -x * initial_momentum / speed2)) : 0.0;
    const double minimum_x = hypot(x + nearest_drift * initial_momentum,
        nearest_drift * tangential_momentum);
    const bool possible_inner = options.inner_boundary == 0
        && start_scale * minimum_x <= edge[0];
    const bool possible_outer = options.outer_boundary == 0
        && end_scale * fmax(x, new_x) >= edge[cells];
    // The interval test selects only paths that may hit a reflecting wall.
    // Resolve each impact time and finish the remaining drift after reflection.
    if (possible_inner || possible_outer) {
        radius = markers.r[i];
        velocity = initial_momentum / start_scale;
        if (!ev::advance_central_drift_with_walls(
                radius, velocity, angular_momentum, dt, drift_tau,
                expansion_alpha, edge[0], options.inner_boundary == 0,
                edge[cells], options.outer_boundary == 0)) {
            atomicExch(error_flag, 12);
            return;
        }
    }
    // Outflow changes the boundary state once a particle crosses the outer
    // face; its mass remains available to the escaped-particle accounting.
    if (options.outer_boundary == 1 && radius > edge[cells]) {
        markers.state[i] = static_cast<std::uint8_t>(MarkerState::escaped);
    }
    markers.r[i] = radius;
    markers.w[i] = velocity;
}

// Apply the gravitational second half kick in the newly solved endpoint
// field. This changes radial velocity while leaving radius and eta fixed.
__global__ void finish_kdk_kernel(
    MarkerDeviceView markers,
    const double* edge,
    const double* radius_center,
    const double* source_mass_edge,
    const double* line_density,
    const double* phi_r,
    int cells,
    DeviceVPOptions options,
    double dt,
    double end_scale,
    int* error_flag) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= markers.size
        || markers.state[i] != static_cast<std::uint8_t>(MarkerState::active)) {
        return;
    }
    const double force = sample_force(
        markers.r[i], edge, radius_center, source_mass_edge, line_density,
        phi_r, cells, options);
    const double momentum = end_scale * markers.w[i]
        - 0.5 * dt * end_scale * force;
    markers.w[i] = momentum / end_scale;
    if (!isfinite(markers.w[i])) {
        atomicExch(error_flag, 13);
    }
}

// Normalize a(t)=[(t+t0)/t0]^alpha at t=0 and require a positive physical
// background time before converting between physical and comoving variables.
inline double scale_factor(double time, const Expansion& expansion) {
    if (!expansion.enabled) {
        return 1.0;
    }
    const double tau = time + expansion.time_offset;
    if (!std::isfinite(time) || !std::isfinite(expansion.alpha)
        || !(expansion.time_offset > 0.0)
        || !std::isfinite(expansion.time_offset) || !(tau > 0.0)
        || !std::isfinite(tau)) {
        throw std::runtime_error("Expanding VP requires positive background time.");
    }
    const double scale = std::pow(
        tau / expansion.time_offset, expansion.alpha);
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        throw std::runtime_error("Expanding VP produced an invalid scale factor.");
    }
    return scale;
}

// The selected expanding background has rho_bar=3*H^2/(8*pi*G). A
// density-contrast input disables this subtraction through its options.
inline double background_density(
    double time, const VlasovPoissonOptions& options) {
    if (!options.expansion.enabled
        || !options.subtract_homogeneous_background) {
        return 0.0;
    }
    const double tau = time + options.expansion.time_offset;
    const double hubble = options.expansion.alpha / tau;
    return 3.0 * hubble * hubble / (8.0 * pi * options.gravity);
}

// Integrate dt/a(t)^2 for the exact free drift in canonical momentum.
// The logarithmic alpha=1/2 limit avoids division by a vanishing exponent.
inline double inverse_scale_squared_integral(
    double start_time,
    double end_time,
    const Expansion& expansion) {
    if (!expansion.enabled || end_time == start_time) {
        return end_time - start_time;
    }
    const double start_tau = start_time + expansion.time_offset;
    const double end_tau = end_time + expansion.time_offset;
    const double start_scale = scale_factor(start_time, expansion);
    const double exponent = 1.0 - 2.0 * expansion.alpha;
    const double log_ratio = std::log(end_tau / start_tau);
    const double prefactor = start_tau / (start_scale * start_scale);
    if (std::abs(exponent)
        < 64.0 * std::numeric_limits<double>::epsilon()) {
        return prefactor * log_ratio;
    }
    return prefactor * std::expm1(exponent * log_ratio) / exponent;
}

// Validate model enums, force parameters, and the positive expansion time
// on the host before passing the compact options into a CUDA kernel.
DeviceVPOptions make_device_options(
    double time, const VlasovPoissonOptions& options) {
    const bool valid_shape = options.particle_shape == ParticleShape::cic
        || options.particle_shape == ParticleShape::tsc;
    const bool valid_inner =
        options.inner_boundary == InnerBoundary::reflecting_wall
        || options.inner_boundary == InnerBoundary::origin_parity;
    const bool valid_outer =
        options.outer_boundary == OuterBoundary::reflecting_wall
        || options.outer_boundary == OuterBoundary::outflow;
    const bool valid_expansion = !options.expansion.enabled
        || (std::isfinite(time) && options.expansion.alpha >= 0.0
            && std::isfinite(options.expansion.alpha)
            && options.expansion.time_offset > 0.0
            && std::isfinite(options.expansion.time_offset)
            && time + options.expansion.time_offset > 0.0
            && std::isfinite(time + options.expansion.time_offset)
            && (!options.subtract_homogeneous_background
                || options.gravity > 0.0));
    if (!std::isfinite(time) || !valid_shape || !valid_inner || !valid_outer
        || !valid_expansion || !(options.gravity >= 0.0)
        || !std::isfinite(options.gravity)
        || !(options.central_mass >= 0.0)
        || !std::isfinite(options.central_mass)) {
        throw std::runtime_error("Invalid Vlasov-Poisson options.");
    }
    DeviceVPOptions result;
    result.gravity = options.gravity;
    result.central_mass = options.central_mass;
    result.rho_bar = background_density(time, options);
    result.particle_shape = static_cast<int>(options.particle_shape);
    result.inner_boundary = static_cast<int>(options.inner_boundary);
    result.outer_boundary = static_cast<int>(options.outer_boundary);
    result.subtract_background = options.subtract_homogeneous_background ? 1 : 0;
    result.carry_inside = options.carry_active_mass_inside_inner_face ? 1 : 0;
    return result;
}

}  // namespace

using detail::DeviceArray;

// Own one common mesh and persistent extensive-moment/source/field arrays.
// The four-entry ledger is a small synchronization result from each deposit.
struct VlasovPoissonModule::Impl {
    explicit Impl(const RadialMesh& host_mesh)
        : mesh(host_mesh),
          edge(host_mesh.edge.size()),
          q_center(host_mesh.cells()),
          radius_center(host_mesh.cells()),
          shell_volume(host_mesh.cells()),
          shell_mass(host_mesh.cells()),
          shell_momentum(host_mesh.cells()),
          shell_radial_second(host_mesh.cells()),
          shell_tangential_second(host_mesh.cells()),
          shell_energy_flux(host_mesh.cells()),
          shell_mass_square(host_mesh.cells()),
          effective_number(host_mesh.cells()),
          source_shell(host_mesh.cells()),
          source_mass_edge(host_mesh.cells() + 1),
          line_density(host_mesh.cells()),
          phi(host_mesh.cells()),
          phi_r(host_mesh.cells()),
          phi_edge(host_mesh.cells() + 1),
          force_integral(host_mesh.cells()),
          force_integral_suffix(host_mesh.cells()),
          ledger(4),
          error_flag(1) {
        edge.upload(mesh.edge.data(), mesh.edge.size());
        q_center.upload(mesh.q_center.data(), mesh.q_center.size());
        radius_center.upload(
            mesh.radius_center.data(), mesh.radius_center.size());
        shell_volume.upload(mesh.shell_volume.data(), mesh.shell_volume.size());
    }

    RadialMesh mesh;
    DeviceArray<double> edge;
    DeviceArray<double> q_center;
    DeviceArray<double> radius_center;
    DeviceArray<double> shell_volume;
    DeviceArray<double> shell_mass;
    DeviceArray<double> shell_momentum;
    DeviceArray<double> shell_radial_second;
    DeviceArray<double> shell_tangential_second;
    DeviceArray<double> shell_energy_flux;
    DeviceArray<double> shell_mass_square;
    DeviceArray<double> effective_number;
    DeviceArray<double> source_shell;
    DeviceArray<double> source_mass_edge;
    DeviceArray<double> line_density;
    DeviceArray<double> phi;
    DeviceArray<double> phi_r;
    DeviceArray<double> phi_edge;
    DeviceArray<double> force_integral;
    DeviceArray<double> force_integral_suffix;
    DeviceArray<double> ledger;
    DeviceArray<int> error_flag;
    DepositionDiagnostics diagnostics{};
};

VlasovPoissonModule::VlasovPoissonModule(const RadialMesh& mesh)
    : impl_(new Impl(mesh)) {}

VlasovPoissonModule::~VlasovPoissonModule() = default;
VlasovPoissonModule::VlasovPoissonModule(VlasovPoissonModule&&) noexcept = default;
VlasovPoissonModule& VlasovPoissonModule::operator=(
    VlasovPoissonModule&&) noexcept = default;

// Host orchestration of one field solve: clear persistent accumulators,
// deposit marker moments, scan to enclosed mass, integrate force/potential,
// and finally download only the four-scalar mass ledger.  The full radial
// field stays device-resident until download_field() is explicitly requested.
DepositionDiagnostics VlasovPoissonModule::deposit_and_solve(
    const WpicState& state,
    double time,
    const VlasovPoissonOptions& options) {
    const int cells = static_cast<int>(impl_->mesh.cells());
    const DeviceVPOptions device_options = make_device_options(time, options);
    impl_->shell_mass.zero();
    impl_->shell_momentum.zero();
    impl_->shell_radial_second.zero();
    impl_->shell_tangential_second.zero();
    impl_->shell_energy_flux.zero();
    impl_->shell_mass_square.zero();
    impl_->ledger.zero();
    impl_->error_flag.zero();

    const MarkerDeviceView markers = state.device_view();
    if (markers.size != 0) {
        const int blocks = static_cast<int>(
            (markers.size + threads_per_block - 1) / threads_per_block);
        deposit_kernel<<<blocks, threads_per_block>>>(
            markers, impl_->edge.data(), impl_->q_center.data(),
            impl_->shell_volume.data(), cells, device_options,
            impl_->shell_mass.data(), impl_->shell_momentum.data(),
            impl_->shell_radial_second.data(),
            impl_->shell_tangential_second.data(),
            impl_->shell_energy_flux.data(), impl_->shell_mass_square.data(),
            impl_->ledger.data(), impl_->error_flag.data());
        detail::check_kernel("WPIC deposition kernel launch failed");
    }

    const int cell_blocks = (cells + threads_per_block - 1) / threads_per_block;
    source_shell_kernel<<<cell_blocks, threads_per_block>>>(
        impl_->shell_mass.data(), impl_->edge.data(),
        impl_->radius_center.data(), cells, device_options,
        impl_->source_shell.data(), impl_->line_density.data());
    detail::check_kernel("Poisson source kernel launch failed");

    // A forward inclusive sum converts shell source masses to face-enclosed
    // masses. Add the unresolved/point contribution after this shell scan.
    thrust::device_ptr<double> source_begin(
        impl_->source_shell.data());
    thrust::device_ptr<double> edge_mass_begin(
        impl_->source_mass_edge.data() + 1);
    thrust::inclusive_scan(source_begin, source_begin + cells, edge_mass_begin);
    add_central_mass_kernel<<<
        (cells + 1 + threads_per_block - 1) / threads_per_block,
        threads_per_block>>>(
        impl_->source_mass_edge.data(), cells,
        options.central_mass - (device_options.subtract_background
            ? (4.0 * pi / 3.0) * device_options.rho_bar
                * impl_->mesh.edge.front() * impl_->mesh.edge.front()
                * impl_->mesh.edge.front() : 0.0),
        impl_->ledger.data());
    detail::check_kernel("Poisson enclosed-mass kernel launch failed");

    field_and_integral_kernel<<<cell_blocks, threads_per_block>>>(
        impl_->edge.data(), impl_->radius_center.data(),
        impl_->source_mass_edge.data(), impl_->line_density.data(), cells,
        options.gravity, impl_->phi_r.data(), impl_->force_integral.data());
    // An inward suffix sum integrates the radial force from the exterior
    // boundary. Potential uses that same analytic force in each cell.
    detail::check_kernel("Poisson force kernel launch failed");

    thrust::device_ptr<double> integral_begin(impl_->force_integral.data());
    thrust::device_ptr<double> suffix_begin(
        impl_->force_integral_suffix.data());
    thrust::inclusive_scan(
        thrust::make_reverse_iterator(integral_begin + cells),
        thrust::make_reverse_iterator(integral_begin),
        thrust::make_reverse_iterator(suffix_begin + cells));
    potential_kernel<<<
        (cells + 1 + threads_per_block - 1) / threads_per_block,
        threads_per_block>>>(
        impl_->edge.data(), impl_->radius_center.data(),
        impl_->source_mass_edge.data(), impl_->line_density.data(),
        impl_->force_integral_suffix.data(), cells, options.gravity,
        impl_->phi_edge.data(), impl_->phi.data());
    derived_moment_kernel<<<cell_blocks, threads_per_block>>>(
        impl_->shell_mass.data(), impl_->shell_mass_square.data(), cells,
        impl_->effective_number.data());
    detail::check_kernel("Poisson potential/moment kernel launch failed");

    // The synchronizing error and ledger copies close asynchronous
    // deposition before publishing its source-mass accounting on the host.
    int error = 0;
    impl_->error_flag.download(&error, 1);
    if (error != 0) {
        throw std::runtime_error(
            "GPU WPIC deposition rejected an invalid marker or mesh stencil; "
            "device error code=" + std::to_string(error));
    }
    double ledger[4] = {0.0, 0.0, 0.0, 0.0};
    impl_->ledger.download(ledger, 4);
    impl_->diagnostics.resolved_active_mass = ledger[0];
    impl_->diagnostics.deposited_mesh_mass = ledger[1];
    impl_->diagnostics.hidden_inner_mass = ledger[2];
    impl_->diagnostics.escaped_mass = ledger[3];
    impl_->diagnostics.mass_roundoff = ledger[1] - ledger[0];
    return impl_->diagnostics;
}

// Complete the self-consistent split: incoming field -> half kick/drift ->
// endpoint field -> half kick. The returned ledger describes endpoint mass;
// velocity moments cached by that deposit precede the final kick. A caller
// writing completed velocity diagnostics deposits again after the full step.
DepositionDiagnostics VlasovPoissonModule::advance_kdk(
    WpicState& state,
    double time,
    double dt,
    const VlasovPoissonOptions& options) {
    if (!(dt >= 0.0) || !std::isfinite(time) || !std::isfinite(dt)) {
        throw std::runtime_error("Invalid GPU weighted-PIC KDK timestep.");
    }
    deposit_and_solve(state, time, options);
    begin_kdk_with_current_field(state, time, dt, options);
    const DepositionDiagnostics endpoint =
        deposit_and_solve(state, time + dt, options);
    finish_kdk_with_current_field(state, time + dt, dt, options);
    return endpoint;
}

void VlasovPoissonModule::begin_kdk_with_current_field(
    WpicState& state,
    double time,
    double dt,
    const VlasovPoissonOptions& options) {
    if (!(dt >= 0.0) || !std::isfinite(time) || !std::isfinite(dt)) {
        throw std::runtime_error("Invalid GPU weighted-PIC KDK timestep.");
    }
    if (dt == 0.0 || state.size() == 0) {
        return;
    }
    // Compute both scale factors and the integrated drift coefficient once
    // for the whole step, then use them for every active characteristic.
    const double start_scale = scale_factor(time, options.expansion);
    const double end_scale = scale_factor(time + dt, options.expansion);
    const double drift_tau = options.expansion.enabled
        ? time + options.expansion.time_offset : 1.0 + dt;
    const double expansion_alpha = options.expansion.enabled
        ? options.expansion.alpha : 0.0;
    const double drift = inverse_scale_squared_integral(
        time, time + dt, options.expansion);
    const DeviceVPOptions device_options = make_device_options(time, options);
    const MarkerDeviceView markers = state.device_view();
    const int blocks = static_cast<int>(
        (markers.size + threads_per_block - 1) / threads_per_block);
    impl_->error_flag.zero();
    begin_kdk_kernel<<<blocks, threads_per_block>>>(
        markers, impl_->edge.data(), impl_->radius_center.data(),
        impl_->source_mass_edge.data(), impl_->line_density.data(),
        impl_->phi_r.data(), static_cast<int>(impl_->mesh.cells()),
        device_options, dt, start_scale, end_scale, drift_tau, expansion_alpha, drift,
        impl_->error_flag.data());
    detail::check_kernel("WPIC KDK begin kernel launch failed");
    int error = 0;
    impl_->error_flag.download(&error, 1);
    if (error != 0) {
        throw std::runtime_error(
            "GPU WPIC KDK drift failed; device error code="
            + std::to_string(error));
    }
}

void VlasovPoissonModule::finish_kdk_with_current_field(
    WpicState& state,
    double endpoint_time,
    double dt,
    const VlasovPoissonOptions& options) {
    if (!(dt >= 0.0) || !std::isfinite(endpoint_time)
        || !std::isfinite(dt)) {
        throw std::runtime_error("Invalid GPU weighted-PIC endpoint kick.");
    }
    if (dt == 0.0 || state.size() == 0) {
        return;
    }
    // The final kick uses background factors and the field at endpoint_time,
    // consistent with the second half of the split Hamiltonian update.
    const double end_scale = scale_factor(endpoint_time, options.expansion);
    const DeviceVPOptions endpoint_options =
        make_device_options(endpoint_time, options);
    const MarkerDeviceView markers = state.device_view();
    const int blocks = static_cast<int>(
        (markers.size + threads_per_block - 1) / threads_per_block);
    impl_->error_flag.zero();
    finish_kdk_kernel<<<blocks, threads_per_block>>>(
        markers, impl_->edge.data(), impl_->radius_center.data(),
        impl_->source_mass_edge.data(), impl_->line_density.data(),
        impl_->phi_r.data(), static_cast<int>(impl_->mesh.cells()),
        endpoint_options, dt, end_scale, impl_->error_flag.data());
    detail::check_kernel("WPIC KDK finish kernel launch failed");
    int error = 0;
    impl_->error_flag.download(&error, 1);
    if (error != 0) {
        throw std::runtime_error(
            "GPU WPIC KDK endpoint kick failed; device error code="
            + std::to_string(error));
    }
}

// MPI uses this path after reducing rank-local extensive moments.  Upload the
// common moments and rebuild the field locally on every GPU, rather than
// communicating particle arrays or a device-specific field representation.
void VlasovPoissonModule::solve_from_global_moments(
    const FieldHost& global,
    double time,
    const VlasovPoissonOptions& options) {
    // All extensive moments must use the common global cell count before
    // upload; each rank then repeats the identical local Poisson construction.
    const std::size_t count = impl_->mesh.cells();
    if (global.shell_mass.size() != count
        || global.shell_radial_momentum.size() != count
        || global.shell_radial_second_moment.size() != count
        || global.shell_tangential_second_moment.size() != count
        || global.shell_radial_energy_flux.size() != count
        || global.shell_mass_square.size() != count) {
        throw std::runtime_error("Global MPI WPIC moment arrays do not match the mesh.");
    }
    impl_->shell_mass.upload(global.shell_mass.data(), count);
    impl_->shell_momentum.upload(global.shell_radial_momentum.data(), count);
    impl_->shell_radial_second.upload(
        global.shell_radial_second_moment.data(), count);
    impl_->shell_tangential_second.upload(
        global.shell_tangential_second_moment.data(), count);
    impl_->shell_energy_flux.upload(
        global.shell_radial_energy_flux.data(), count);
    impl_->shell_mass_square.upload(global.shell_mass_square.data(), count);
    const double ledger[4] = {
        global.diagnostics.resolved_active_mass,
        global.diagnostics.deposited_mesh_mass,
        global.diagnostics.hidden_inner_mass,
        global.diagnostics.escaped_mass,
    };
    impl_->ledger.upload(ledger, 4);
    impl_->diagnostics = global.diagnostics;

    const int cells = static_cast<int>(count);
    const DeviceVPOptions device_options = make_device_options(time, options);
    const int blocks = (cells + threads_per_block - 1) / threads_per_block;
    source_shell_kernel<<<blocks, threads_per_block>>>(
        impl_->shell_mass.data(), impl_->edge.data(),
        impl_->radius_center.data(), cells, device_options,
        impl_->source_shell.data(), impl_->line_density.data());
    detail::check_kernel("Global Poisson source kernel launch failed");
    thrust::device_ptr<double> source_begin(impl_->source_shell.data());
    thrust::device_ptr<double> edge_mass_begin(
        impl_->source_mass_edge.data() + 1);
    thrust::inclusive_scan(source_begin, source_begin + cells, edge_mass_begin);
    add_central_mass_kernel<<<
        (cells + 1 + threads_per_block - 1) / threads_per_block,
        threads_per_block>>>(
        impl_->source_mass_edge.data(), cells,
        options.central_mass - (device_options.subtract_background
            ? (4.0 * pi / 3.0) * device_options.rho_bar
                * impl_->mesh.edge.front() * impl_->mesh.edge.front()
                * impl_->mesh.edge.front() : 0.0),
        impl_->ledger.data());
    field_and_integral_kernel<<<blocks, threads_per_block>>>(
        impl_->edge.data(), impl_->radius_center.data(),
        impl_->source_mass_edge.data(), impl_->line_density.data(), cells,
        options.gravity, impl_->phi_r.data(), impl_->force_integral.data());
    detail::check_kernel("Global Poisson force kernel launch failed");
    thrust::device_ptr<double> integral_begin(impl_->force_integral.data());
    thrust::device_ptr<double> suffix_begin(
        impl_->force_integral_suffix.data());
    thrust::inclusive_scan(
        thrust::make_reverse_iterator(integral_begin + cells),
        thrust::make_reverse_iterator(integral_begin),
        thrust::make_reverse_iterator(suffix_begin + cells));
    potential_kernel<<<
        (cells + 1 + threads_per_block - 1) / threads_per_block,
        threads_per_block>>>(
        impl_->edge.data(), impl_->radius_center.data(),
        impl_->source_mass_edge.data(), impl_->line_density.data(),
        impl_->force_integral_suffix.data(), cells, options.gravity,
        impl_->phi_edge.data(), impl_->phi.data());
    derived_moment_kernel<<<blocks, threads_per_block>>>(
        impl_->shell_mass.data(), impl_->shell_mass_square.data(), cells,
        impl_->effective_number.data());
    detail::check_kernel("Global Poisson potential kernel launch failed");
}

// Explicit output boundary: download the stored shell moments and actual
// source/field arrays, retaining their common mesh dimensions and mass ledger.
FieldHost VlasovPoissonModule::download_field() const {
    const std::size_t cells = impl_->mesh.cells();
    FieldHost host;
    host.shell_mass.resize(cells);
    host.shell_radial_momentum.resize(cells);
    host.shell_radial_second_moment.resize(cells);
    host.shell_tangential_second_moment.resize(cells);
    host.shell_radial_energy_flux.resize(cells);
    host.shell_mass_square.resize(cells);
    host.effective_marker_number.resize(cells);
    host.source_mass_edge.resize(cells + 1);
    host.source_line_density.resize(cells);
    host.phi.resize(cells);
    host.phi_r.resize(cells);
    impl_->shell_mass.download(host.shell_mass.data(), cells);
    impl_->shell_momentum.download(
        host.shell_radial_momentum.data(), cells);
    impl_->shell_radial_second.download(
        host.shell_radial_second_moment.data(), cells);
    impl_->shell_tangential_second.download(
        host.shell_tangential_second_moment.data(), cells);
    impl_->shell_energy_flux.download(
        host.shell_radial_energy_flux.data(), cells);
    impl_->shell_mass_square.download(host.shell_mass_square.data(), cells);
    impl_->effective_number.download(
        host.effective_marker_number.data(), cells);
    impl_->source_mass_edge.download(host.source_mass_edge.data(), cells + 1);
    impl_->line_density.download(host.source_line_density.data(), cells);
    impl_->phi.download(host.phi.data(), cells);
    impl_->phi_r.download(host.phi_r.data(), cells);
    host.diagnostics = impl_->diagnostics;
    return host;
}

// Transfer the already computed scalar face potential needed to normalize
// the uniform-interior receiver field. The device field is left unchanged.
double VlasovPoissonModule::download_inner_face_potential() const {
    double value = 0.;
    VGPU_CUDA_CHECK(cudaMemcpy(&value, impl_->phi_edge.data(),
        sizeof(value), cudaMemcpyDeviceToHost));
    return value;
}

const RadialMesh& VlasovPoissonModule::mesh() const noexcept {
    return impl_->mesh;
}

}  // namespace venus_gpu
