#pragma once

// Local elastic encounters between spherical simulation particles.
// Inputs are continuous (r,w,eta) particles, the radial mesh, physical cross-section
// or Plummer encounter parameters, an elapsed collision interval and a persistent
// random stream. Radius sorting defines collision shells; local three-dimensional
// velocities supply the Boltzmann pair kernel. The simulation-particle masses
// enter the physical pair law. Each accepted event updates both particles and
// preserves their masses, pair momentum and kinetic energy. Shell refinement and local substeps bound the spatial and time
// approximation. The caller separately combines this operator with transport.

#include "ev/spherical_dsmc_collision.hpp"
#include "ev/weighted_vlasov_pic.hpp"

#include <sstream>

namespace ev {

enum class WeightedPICCollisionPhysics {
    venus_cross_section,
    gravitational_plummer,
};

struct WeightedPICCollisionOptions {
    WeightedPICCollisionPhysics physics =
        WeightedPICCollisionPhysics::venus_cross_section;
    std::size_t target_shell_markers = 4096;
    std::size_t minimum_shell_markers = 32;
    // Configuration field retained in the interface contract. The collision
    // operator supports unequal marker masses independently of this value.
    double equal_weight_relative_tolerance =
        64.0 * std::numeric_limits<double>::epsilon();
    double collision_cfl = 1.0;
    int maximum_substeps = 200;
    bool fail_on_nonlocal_shell = false;
    // Reference mode gives every shell the timestep imposed by the fastest
    // shell.  Adaptive mode instead advances each independent shell on its
    // own CFL clock while marker radii are frozen during the split collision
    // operator.  This changes neither the shell collision rate nor the pair
    // acceptance law; it only avoids revisiting slow outer shells at every
    // dense-core substep.
    bool adaptive_shell_timestep = false;
    std::size_t scheduling_pair_samples_per_shell = 128;
    double scheduling_standard_error_buffer = 4.0;
    bool at_most_one_collision_per_marker = false;
    bool radial_cell_shells = false;
    bool input_ordered_by_radius = false;

    // Equal-reference-mass encounters calibrate the aperture per unit mass.
    // gravitational_scatterer_mass is that calibration mass, whereas each
    // marker's represented mass supplies its actual unequal-mass pair scale.
    // b_max is a required physical large-impact cutoff and b_min can exclude
    // unresolved close encounters.
    double gravitational_scatterer_mass = 0.0;
    double gravitational_plummer_pair_softening = 0.0;
    double gravitational_b_min = 0.0;
    double gravitational_b_max = 0.0;
    double gravitational_deflection_tolerance = 5.0e-13;
    int gravitational_deflection_max_depth = 24;
};

struct WeightedPICCollisionShell {
    std::size_t begin = 0;
    std::size_t end = 0;
    double inner_edge = 0.0;
    double outer_edge = 0.0;
    double volume = 0.0;
    double number_density = 0.0;
    double density = 0.0;
    double minimum_mass = std::numeric_limits<double>::infinity();
    double maximum_mass = 0.0;
    double mean_free_path = std::numeric_limits<double>::infinity();
    std::uint64_t mean_free_path_pair_samples = 0;
    double orbital_velocity_dispersion = 0.0;
    double maximum_speed = 0.0;
    double majorant_u_sigma = 0.0;
};

struct WeightedPICCollisionEvent {
    std::size_t first_marker = 0;
    std::size_t second_marker = 0;
    BPFAVec3 first_before{};
    BPFAVec3 second_before{};
    BPFAVec3 first_after{};
    BPFAVec3 second_after{};
};

struct WeightedPICCollisionDiagnostics {
    std::uint64_t substeps = 0;
    std::uint64_t shell_visits = 0;
    std::uint64_t trial_pairs = 0;
    std::uint64_t accepted_pairs = 0;
    std::uint64_t skipped_already_collided_trials = 0;
    std::size_t active_markers = 0;
    std::size_t minimum_shell_population = 0;
    std::size_t nonlocal_shells = 0;
    std::size_t nonlocal_occupied_shells = 0;
    double minimum_mean_free_path = std::numeric_limits<double>::infinity();
    double maximum_shell_width_over_mean_free_path = 0.0;
    double maximum_occupied_shell_span_over_mean_free_path = 0.0;
    std::uint64_t mean_free_path_pair_samples = 0;
    double minimum_orbital_velocity_dispersion =
        std::numeric_limits<double>::infinity();
    double maximum_orbital_velocity_dispersion = 0.0;
    double maximum_relative_weight_spread = 0.0;
    double represented_event_mass = 0.0;
    double radial_momentum_error = 0.0;
    double kinetic_energy_error = 0.0;
    double max_pair_momentum_error = 0.0;
    double max_pair_kinetic_energy_error = 0.0;
};

struct WeightedPICCollisionWorkspace {
    struct ThreadCache {
        std::vector<DSMCVelocity> velocity;
        std::vector<unsigned char> initialized;
        std::vector<unsigned char> changed;
        std::vector<std::size_t> changed_local_indices;
    };

    std::vector<std::size_t> active_by_radius;
    // For a reflecting isolated halo, the active population and marker IDs do
    // not change.  Retaining its ID order avoids validating and rebuilding the
    // same active list, and lets the stable radius radix use eight passes
    // instead of re-sorting the secondary ID key on every event.
    std::vector<std::size_t> active_by_id;
    bool active_population_is_invariant = false;
    std::vector<std::uint64_t> radius_key_by_marker;
    std::vector<std::size_t> radix_scratch;
    std::vector<std::uint64_t> radix_histogram;
    std::vector<std::uint64_t> radix_offsets;
    std::vector<std::vector<std::size_t>> active_by_thread;
    std::vector<std::vector<WeightedPICCollisionShell>> shells_by_block;
    std::vector<WeightedPICCollisionShell> shells;
    std::vector<DSMCCollisionStats> shell_stats;
    std::vector<std::uint64_t> shell_substeps;
    std::vector<std::vector<WeightedPICCollisionEvent>> shell_events;
    std::vector<ThreadCache> thread_cache;
};

namespace weighted_pic_collision_detail {

// At the regular origin a finite spherical distribution has eta=0.  The
// stored |w| is its speed; its direction is isotropic and its mean projection
// onto any fixed Cartesian axis is zero.
// For eta=L^2/2 the squared transverse speed is 2*eta/r^2. Evaluate the
// radial eta=0 limit first so exactly radial particles remain finite at the origin.
inline long double tangential_speed_squared(const WeightedPICMarker& marker) {
    if (marker.eta == 0.0) return 0.0L;
    const long double radius = marker.r;
    const long double squared_radius = radius * radius;
    return squared_radius > 0.0L && std::isfinite(squared_radius)
        ? 2.0L * marker.eta / squared_radius
        : (2.0L * marker.eta / radius) / radius;
}

inline long double mean_radial_velocity(const WeightedPICMarker& marker) {
    return marker.r == 0.0 ? 0.0L : marker.w;
}

inline BPFAVec3 isotropic_origin_velocity(
    const WeightedPICMarker& marker, double cosine, double azimuth) {
    const double speed = std::abs(marker.w);
    const double transverse = speed * std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
    return {speed * cosine, transverse * std::cos(azimuth), transverse * std::sin(azimuth)};
}

inline void validate_collision_marker(
    const WeightedPICMarker& marker, const WeightedPICRadialMesh& mesh) {
    validate_weighted_pic_marker(marker, "Weighted PIC collisions");
    if (marker.state != WeightedPICMarkerState::active) return;
    if (marker.r < mesh.edge.front() || marker.r > mesh.edge.back()) {
        throw std::runtime_error(
            "An active weighted PIC collision marker is outside the radial mesh.");
    }
    if (marker.r == 0.0 && marker.eta != 0.0) {
        throw std::runtime_error(
            "A material collision marker at the origin requires eta=0.");
    }
    if (marker.r > 0.0 && marker.r * marker.r == 0.0) {
        throw std::runtime_error(
            "Material collisions require a representable positive squared radius "
            "to store transverse kinetic energy.");
    }
}

inline void validate_options(
    const WeightedPICCollisionOptions& options,
    const Params& p) {
    if (options.target_shell_markers < 2
        || options.minimum_shell_markers < 2
        || options.minimum_shell_markers > options.target_shell_markers
        || !(options.equal_weight_relative_tolerance >= 0.0)
        || !std::isfinite(options.equal_weight_relative_tolerance)
        || !(options.collision_cfl > 0.0)
        || !std::isfinite(options.collision_cfl)
        || options.maximum_substeps < 1
        || options.scheduling_pair_samples_per_shell < 2
        || !(options.scheduling_standard_error_buffer >= 0.0)
        || !std::isfinite(options.scheduling_standard_error_buffer)) {
        throw std::runtime_error("Invalid weighted PIC collision options.");
    }
    if (options.physics == WeightedPICCollisionPhysics::venus_cross_section) {
        if (!(p.bp_cross_section == "constant"
              || p.bp_cross_section == "powerlaw"
              || p.bp_cross_section == "yukawa_born")) {
            throw std::runtime_error(
                "Weighted PIC collisions support constant, powerlaw, and "
                "yukawa_born VENUS kernels.  Use gravitational_plummer for "
                "finite-angle softened gravitational encounters; the "
                "softened_gravity scalar transport fit is not a differential kernel.");
        }
        if (bp_fa_kernel_name(p) == "yukawa_born"
            && p.bp_cross_section != "yukawa_born") {
            throw std::runtime_error(
                "The Yukawa differential kernel requires bp_cross_section=yukawa_born.");
        }
        if (!(bp_fa_kernel_name(p) == "isotropic"
              || bp_fa_kernel_name(p) == "yukawa_born")) {
            throw std::runtime_error(
                "Weighted PIC collisions received an unsupported differential kernel.");
        }
    } else {
        if (!(options.gravitational_scatterer_mass > 0.0)
            || !std::isfinite(options.gravitational_scatterer_mass)
            || !(options.gravitational_b_min >= 0.0)
            || !std::isfinite(options.gravitational_b_min)
            || !(options.gravitational_b_max > options.gravitational_b_min)
            || !std::isfinite(options.gravitational_b_max)
            || !(options.gravitational_plummer_pair_softening > 0.0)
            || !std::isfinite(options.gravitational_plummer_pair_softening)
            || !(options.gravitational_deflection_tolerance > 0.0)
            || !std::isfinite(options.gravitational_deflection_tolerance)
            || options.gravitational_deflection_max_depth < 4
            || !(p.vp_gravity > 0.0)) {
            throw std::runtime_error(
                "Plummer PIC gravity collisions require positive physical mass, "
                "pair softening, gravity, quadrature controls, and an explicit "
                "finite impact-parameter interval.");
        }
    }
}

inline double total_cross_section_per_mass(
    double relative_speed,
    const Params& p,
    const WeightedPICCollisionOptions& options) {
    if (options.physics == WeightedPICCollisionPhysics::venus_cross_section) {
        return bp_fa_total_cross_section(relative_speed, p);
    }
    const double area = PI
        * (options.gravitational_b_max * options.gravitational_b_max
           - options.gravitational_b_min * options.gravitational_b_min);
    return p.bp_collision_strength * area
        / options.gravitational_scatterer_mass;
}

/// Physical cross section of one unequal-mass pair.
///
/// The material kernels are configured as sigma/m.  The symmetric pair
/// closure sigma_ij=[(m_i+m_j)/2](sigma/m) is a symmetric arithmetic
/// continuation chosen to recover the configured equal-mass collision
/// frequency.  It is a physical coarse-object closure, not a mathematical
/// uniqueness statement about all possible unequal-mass models.
/// For the gravitational aperture, the configured area belongs to the
/// reference scatterer mass.  Its area per reference mass is multiplied by
/// the same pair mean; the deflection itself uses M_i+M_j below.
// The pair cross section multiplies sigma/m by the mean simulation-particle
// mass. This defines the unequal physical-mass model used by this operator.
// Statistical weights for material constituents require their own collision law.
inline double pair_total_cross_section(
    double relative_speed,
    double first_mass,
    double second_mass,
    const Params& p,
    const WeightedPICCollisionOptions& options) {
    if (!(first_mass > 0.0) || !(second_mass > 0.0)
        || !std::isfinite(first_mass) || !std::isfinite(second_mass)) {
        throw std::runtime_error(
            "Weighted PIC collision pair has a non-positive mass.");
    }
    return 0.5 * (first_mass + second_mass)
        * total_cross_section_per_mass(relative_speed, p, options);
}

inline double u_sigma_majorant(
    double u_max,
    const Params& p,
    const WeightedPICCollisionOptions& options) {
    if (!(u_max > 0.0)) {
        return 0.0;
    }
    if (options.physics == WeightedPICCollisionPhysics::venus_cross_section) {
        return dsmc_u_sigma_majorant(u_max, p);
    }
    return u_max * total_cross_section_per_mass(u_max, p, options)
        * (1.0 + 128.0 * std::numeric_limits<double>::epsilon());
}

inline double pair_u_sigma_majorant(
    double u_max,
    double maximum_mass,
    const Params& p,
    const WeightedPICCollisionOptions& options) {
    if (!(u_max > 0.0) || !(maximum_mass > 0.0)) {
        return 0.0;
    }
    return maximum_mass * u_sigma_majorant(u_max, p, options);
}

template <class Function>
inline long double adaptive_simpson_recursive(
    const Function& function,
    long double left,
    long double right,
    long double f_left,
    long double f_middle,
    long double f_right,
    long double whole,
    long double tolerance,
    int depth) {
    const long double middle = 0.5L * (left + right);
    const long double left_middle = 0.5L * (left + middle);
    const long double right_middle = 0.5L * (middle + right);
    const long double f_left_middle = function(left_middle);
    const long double f_right_middle = function(right_middle);
    const long double left_integral = (middle - left)
        * (f_left + 4.0L * f_left_middle + f_middle) / 6.0L;
    const long double right_integral = (right - middle)
        * (f_middle + 4.0L * f_right_middle + f_right) / 6.0L;
    const long double delta = left_integral + right_integral - whole;
    if (depth == 0 || std::abs(delta) <= 15.0L * tolerance) {
        return left_integral + right_integral + delta / 15.0L;
    }
    return adaptive_simpson_recursive(
               function,
               left,
               middle,
               f_left,
               f_left_middle,
               f_middle,
               left_integral,
               0.5L * tolerance,
               depth - 1)
        + adaptive_simpson_recursive(
               function,
               middle,
               right,
               f_middle,
               f_right_middle,
               f_right,
               right_integral,
               0.5L * tolerance,
               depth - 1);
}

// Exact central-force deflection for the symmetric Plummer-softened pair
// potential.  Sampling b in area and mapping it through this function samples
// the complete differential cross section, including both sides of the
// non-monotone deflection function near the soft core.
// For the softened gravitational law, conserve energy and angular momentum
// along the two-body orbit. Integrate its angle from the turning point to
// infinity after removing the turning-point square-root singularity.
inline double plummer_deflection_angle(
    double impact_parameter,
    double relative_speed,
    double first_mass,
    double second_mass,
    const Params& p,
    const WeightedPICCollisionOptions& options) {
    if (!(impact_parameter > 0.0) || !(relative_speed > 0.0)) {
        return 0.0;
    }
    // The configured aperture area per reference mass is held fixed.  Its
    // associated length scale is therefore proportional to sqrt(pair mean
    // mass).  Apply the same scale to the Plummer softening so the sampled
    // aperture and the softened encounter geometry remain homologous.
    const long double reference_mass =
        options.gravitational_scatterer_mass;
    const long double pair_length_scale = std::sqrt(
        (static_cast<long double>(first_mass)
         + static_cast<long double>(second_mass))
        / (2.0L * reference_mass));
    const long double epsilon =
        options.gravitational_plummer_pair_softening
        * pair_length_scale;
    const long double scaled_b = impact_parameter / epsilon;
    const long double gravitational_length =
        static_cast<long double>(p.vp_gravity)
        * (static_cast<long double>(first_mass)
           + static_cast<long double>(second_mass))
        / (static_cast<long double>(relative_speed) * relative_speed);
    const long double scaled_a = gravitational_length / epsilon;

    // The incoming positive-energy orbit has one centrifugal turning point in
    // (0,b).  Bisection is deterministic and remains robust through the
    // rainbow part of the softened deflection function.
    auto radial_function = [&](long double radius) {
        const long double ratio = scaled_b / radius;
        return 1.0L - ratio * ratio
            + 2.0L * scaled_a / std::sqrt(radius * radius + 1.0L);
    };
    long double lower = 0.0L;
    long double upper = scaled_b;
    for (int iteration = 0; iteration < 128; ++iteration) {
        const long double middle = 0.5L * (lower + upper);
        if (radial_function(middle) > 0.0L) {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    const long double turning = 0.5L * (lower + upper);
    const long double turning_norm = std::sqrt(turning * turning + 1.0L);

    // With x=r_min/r=1-t^2, the square-root endpoint singularity cancels
    // analytically.  Factoring F=(1-x)H also avoids cancellation near r_min.
    auto regular_integrand = [&](long double t) {
        const long double x = 1.0L - t * t;
        const long double x2 = x * x;
        const long double local_norm = std::sqrt(turning * turning + x2);
        const long double numerator =
            turning * turning * (1.0L + x)
            + (1.0L + x + x2 + x2 * x);
        const long double denominator = local_norm * turning_norm
            * (turning_norm + x * local_norm);
        const long double h = 1.0L + x
            + 2.0L * scaled_a * x * numerator / denominator;
        return 2.0L / std::sqrt(h);
    };
    const long double f_left = regular_integrand(0.0L);
    const long double f_middle = regular_integrand(0.5L);
    const long double f_right = regular_integrand(1.0L);
    const long double whole =
        (f_left + 4.0L * f_middle + f_right) / 6.0L;
    const long double tolerance =
        options.gravitational_deflection_tolerance
        * std::max(1.0L, std::abs(whole));
    const long double integral = adaptive_simpson_recursive(
        regular_integrand,
        0.0L,
        1.0L,
        f_left,
        f_middle,
        f_right,
        whole,
        tolerance,
        options.gravitational_deflection_max_depth);
    const long double half_orbit_angle = scaled_b * integral / turning;
    return static_cast<double>(std::clamp(
        std::abs(2.0L * half_orbit_angle
                 - static_cast<long double>(PI)),
        0.0L,
        static_cast<long double>(PI)));
}

// Equal-reference-mass overload used by canonical
// collision diagnostics and the equal-mass regression cases.
inline double plummer_deflection_angle(
    double impact_parameter,
    double relative_speed,
    const Params& p,
    const WeightedPICCollisionOptions& options) {
    return plummer_deflection_angle(
        impact_parameter,
        relative_speed,
        options.gravitational_scatterer_mass,
        options.gravitational_scatterer_mass,
        p,
        options);
}

// Uniform incident area means b^2 is uniform over the physical aperture.
// Mapping that draw through the orbit deflection supplies its full angular law.
inline double sample_impact_parameter_from_unit(
    const WeightedPICCollisionOptions& options,
    double unit) {
    const double b2_min = options.gravitational_b_min
        * options.gravitational_b_min;
    const double b2_max = options.gravitational_b_max
        * options.gravitational_b_max;
    return std::sqrt(
        b2_min + std::clamp(unit, 0.0, 1.0) * (b2_max - b2_min));
}

inline double sample_impact_parameter(
    const WeightedPICCollisionOptions& options,
    DSMCSplitMix64& rng) {
    return sample_impact_parameter_from_unit(options, rng.uniform_open());
}

inline double sample_scatter_mu_from_unit(
    double relative_speed,
    double first_mass,
    double second_mass,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    double unit) {
    const double clipped = std::clamp(unit, 0.0, 1.0);
    if (options.physics == WeightedPICCollisionPhysics::venus_cross_section) {
        if (bp_fa_kernel_name(p) != "yukawa_born") {
            return std::fma(2.0, clipped, -1.0);
        }
        return bp_fa_yukawa_born_mu_from_unit(relative_speed, p, clipped);
    }
    const double pair_length_scale = std::sqrt(
        0.5 * (first_mass + second_mass)
        / options.gravitational_scatterer_mass);
    const double impact_parameter = pair_length_scale
        * sample_impact_parameter_from_unit(options, clipped);
    const double theta = plummer_deflection_angle(
        impact_parameter,
        relative_speed,
        first_mass,
        second_mass,
        p,
        options);
    return std::cos(theta);
}

inline double sample_scatter_mu_from_unit(
    double relative_speed,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    double unit) {
    return sample_scatter_mu_from_unit(
        relative_speed,
        options.gravitational_scatterer_mass,
        options.gravitational_scatterer_mass,
        p,
        options,
        unit);
}

inline double sample_scatter_mu(
    double relative_speed,
    double first_mass,
    double second_mass,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    DSMCSplitMix64& rng) {
    return sample_scatter_mu_from_unit(
        relative_speed,
        first_mass,
        second_mass,
        p,
        options,
        rng.uniform_open());
}

inline std::uint64_t collision_seed(
    std::uint64_t base,
    std::uint64_t counter,
    std::uint64_t stream) {
    return weighted_pic_mix64(
        base ^ weighted_pic_mix64(counter + UINT64_C(0x9e3779b97f4a7c15))
        ^ weighted_pic_mix64(stream + UINT64_C(0xd1b54a32d192ed03)));
}

inline double radius_between(double first, double second) {
    return std::cbrt(0.5 * (first * first * first + second * second * second));
}

inline double summarize_marker_masses(
    const std::vector<WeightedPICMarker>& markers,
    const std::vector<std::size_t>& active,
    double& relative_spread) {
    if (active.empty()) {
        relative_spread = 0.0;
        return 0.0;
    }
    double total = 0.0;
    double minimum = markers[active.front()].mass;
    double maximum = minimum;
    for (const std::size_t index : active) {
        const double mass = markers[index].mass;
        total += mass;
        minimum = std::min(minimum, mass);
        maximum = std::max(maximum, mass);
    }
    const double mean = total / static_cast<double>(active.size());
    relative_spread = (maximum - minimum) / mean;
    return mean;
}

struct MeanFreePathEstimate {
    double mean_free_path = std::numeric_limits<double>::infinity();
    double orbital_velocity_dispersion = 0.0;
    std::uint64_t pair_samples = 0;
};

/// Frank-compatible mean free path for the current shell distribution.
///
/// For complete unequal-mass markers with pair cross section Sigma_ij(u), the
/// distance travelled per collision is
///
///   lambda_mfp = <u> / [n_partner <u Sigma_ij(u)>],
///
/// with n_partner=(N-1)/V in a finite shell.  This matches the unordered-pair
/// trial law exactly; n=N/V is the large-population approximation.
///
/// The deterministic pair sample includes the unresolved relative tangential
/// azimuth.  Consequently both the effective velocity-dependent cross section
/// and the orbital velocity dispersion follow the evolving distribution
/// function.  Equal masses and constant sigma/m approach Frank & Tokayer's
/// lambda_mfp=1/[rho*(sigma/m)] with the finite-N factor N/(N-1).
// Use the finite partner density (N-1)/V and the same u*Sigma pair law as
// actual encounters. The distance per event is <u>/[(N-1)*<u*Sigma>/V].
inline MeanFreePathEstimate velocity_averaged_mean_free_path(
    const std::vector<WeightedPICMarker>& markers,
    const std::vector<std::size_t>& order,
    std::size_t begin,
    std::size_t end,
    double volume,
    const Params& p,
    const WeightedPICCollisionOptions& options) {
    MeanFreePathEstimate result;
    const std::size_t count = end - begin;
    if (count < 2 || !(volume > 0.0)) {
        return result;
    }

    long double mass_sum = 0.0L;
    long double radial_momentum = 0.0L;
    long double speed_squared_moment = 0.0L;
    double minimum_mass = std::numeric_limits<double>::infinity();
    double maximum_mass = 0.0;
    for (std::size_t local = begin; local < end; ++local) {
        const WeightedPICMarker& marker = markers[order[local]];
        const long double mass = marker.mass;
        const long double tangential_speed_squared =
            weighted_pic_collision_detail::tangential_speed_squared(marker);
        mass_sum += mass;
        radial_momentum += mass * mean_radial_velocity(marker);
        speed_squared_moment += mass
            * (marker.w * marker.w + tangential_speed_squared);
        minimum_mass = std::min(minimum_mass, marker.mass);
        maximum_mass = std::max(maximum_mass, marker.mass);
    }
    const long double mean_radial_velocity = radial_momentum / mass_sum;
    const long double centred_three_dimensional_variance = std::max(
        0.0L,
        speed_squared_moment / mass_sum
            - mean_radial_velocity * mean_radial_velocity);
    result.orbital_velocity_dispersion = std::sqrt(
        static_cast<double>(centred_three_dimensional_variance / 3.0L));

    const long double pair_count =
        0.5L * static_cast<long double>(count)
        * static_cast<long double>(count - 1);
    if (options.physics == WeightedPICCollisionPhysics::venus_cross_section
        && p.bp_cross_section == "constant"
        && minimum_mass == maximum_mass) {
        // For equal masses and a velocity-independent cross section, the
        // factor of relative speed cancels analytically between <u> and
        // <u Sigma>.  Every pair is therefore represented exactly without
        // repeating the trigonometric pair pilot used by general kernels.
        const double pair_cross_section = pair_total_cross_section(
            1.0, minimum_mass, minimum_mass, p, options);
        const double partner_number_density =
            static_cast<double>(count - 1) / volume;
        if (pair_cross_section > 0.0) {
            result.mean_free_path =
                1.0 / (partner_number_density * pair_cross_section);
        }
        result.pair_samples = static_cast<std::uint64_t>(pair_count);
        return result;
    }
    constexpr std::uint64_t requested_samples = UINT64_C(512);
    result.pair_samples = static_cast<std::uint64_t>(std::min(
        pair_count,
        static_cast<long double>(requested_samples)));
    if (result.pair_samples == 0) {
        return result;
    }

    const std::uint64_t seed =
        UINT64_C(0x6d66705f77706963)
        ^ markers[order[begin]].id
        ^ (markers[order[end - 1]].id + UINT64_C(0x9e3779b97f4a7c15))
        ^ static_cast<std::uint64_t>(begin)
        ^ (static_cast<std::uint64_t>(end) << 32);
    long double relative_speed_sum = 0.0L;
    long double collision_kernel_sum = 0.0L;
    auto sample_pair = [&](std::size_t first_local,
                           std::size_t second_local,
                           std::uint64_t sample) {
        const WeightedPICMarker& first = markers[order[begin + first_local]];
        const WeightedPICMarker& second = markers[order[begin + second_local]];
        const double relative_azimuth =
            2.0 * PI * dsmc_splitmix_uniform_open_at(seed, 3 * sample + 2);
        const double first_tangential_speed = std::sqrt(2.0 * first.eta)
            / std::max(first.r, 1.0e-300);
        const double second_tangential_speed = std::sqrt(2.0 * second.eta)
            / std::max(second.r, 1.0e-300);
        const double radial_difference = first.w - second.w;
        double relative_speed = std::sqrt(std::max(
            0.0,
            radial_difference * radial_difference
                + first_tangential_speed * first_tangential_speed
                + second_tangential_speed * second_tangential_speed
                - 2.0 * first_tangential_speed
                    * second_tangential_speed * std::cos(relative_azimuth)));
        if (first.r == 0.0 || second.r == 0.0) {
            BPFAVec3 first_velocity{first.w, first_tangential_speed, 0.0};
            BPFAVec3 second_velocity{second.w,
                second_tangential_speed * std::cos(relative_azimuth),
                second_tangential_speed * std::sin(relative_azimuth)};
            const auto origin_draw = [&](std::uint64_t dimension) {
                return dsmc_splitmix_uniform_open_at(
                    seed ^ UINT64_C(0x6f726967696e5f76), 4 * sample + dimension);
            };
            if (first.r == 0.0) first_velocity = isotropic_origin_velocity(
                first, 2.0 * origin_draw(0) - 1.0, 2.0 * PI * origin_draw(1));
            if (second.r == 0.0) second_velocity = isotropic_origin_velocity(
                second, 2.0 * origin_draw(2) - 1.0, 2.0 * PI * origin_draw(3));
            relative_speed = bp_fa_norm(bp_fa_sub(first_velocity, second_velocity));
        }
        relative_speed_sum += relative_speed;
        collision_kernel_sum += relative_speed
            * pair_total_cross_section(
                relative_speed, first.mass, second.mass, p, options);
    };

    if (pair_count <= static_cast<long double>(requested_samples)) {
        std::uint64_t sample = 0;
        for (std::size_t first = 0; first < count; ++first) {
            for (std::size_t second = first + 1;
                 second < count;
                 ++second) {
                sample_pair(first, second, sample++);
            }
        }
    } else {
        for (std::uint64_t sample = 0;
             sample < result.pair_samples;
             ++sample) {
            const std::size_t first = std::min(
                count - 1,
                static_cast<std::size_t>(
                    dsmc_splitmix_uniform_open_at(seed, 3 * sample)
                    * static_cast<double>(count)));
            std::size_t second = std::min(
                count - 2,
                static_cast<std::size_t>(
                    dsmc_splitmix_uniform_open_at(seed, 3 * sample + 1)
                    * static_cast<double>(count - 1)));
            if (second >= first) {
                ++second;
            }
            sample_pair(first, second, sample);
        }
    }

    const long double partner_number_density =
        static_cast<long double>(count - 1) / volume;
    if (relative_speed_sum > 0.0L && collision_kernel_sum > 0.0L) {
        result.mean_free_path = static_cast<double>(
            relative_speed_sum
            / (partner_number_density * collision_kernel_sum));
    }
    return result;
}

// A shell has physical volume between radius-sorted neighbours. Its actual
// mass/velocity moments supply density, dispersion, mean free path and the
// maximum pair-rate envelope used by the local sampler.
inline WeightedPICCollisionShell describe_shell(
    const std::vector<WeightedPICMarker>& markers,
    const std::vector<std::size_t>& order,
    const WeightedPICRadialMesh& mesh,
    double marker_mass,
    std::size_t begin,
    std::size_t end,
    const Params& p,
    const WeightedPICCollisionOptions& options) {
    WeightedPICCollisionShell shell;
    shell.begin = begin;
    shell.end = end;
    const double first_radius = markers[order[begin]].r;
    const double last_radius = markers[order[end - 1]].r;
    shell.inner_edge = begin == 0
        ? mesh.edge.front()
        : radius_between(markers[order[begin - 1]].r, first_radius);
    shell.outer_edge = end == order.size()
        ? mesh.edge.back()
        : radius_between(last_radius, markers[order[end]].r);
    if (!(shell.outer_edge > shell.inner_edge)) {
        throw std::runtime_error(
            "Weighted PIC collision shell has non-positive spherical volume.");
    }
    shell.volume = (4.0 * PI / 3.0)
        * (std::pow(shell.outer_edge, 3) - std::pow(shell.inner_edge, 3));
    (void)marker_mass;
    long double shell_mass = 0.0L;
    long double radial_momentum = 0.0L;
    long double speed_squared_moment = 0.0L;
    for (std::size_t local = begin; local < end; ++local) {
        const WeightedPICMarker& marker = markers[order[local]];
        shell_mass += marker.mass;
        radial_momentum +=
            static_cast<long double>(marker.mass) * mean_radial_velocity(marker);
        const long double tangential_speed_squared =
            weighted_pic_collision_detail::tangential_speed_squared(marker);
        speed_squared_moment += static_cast<long double>(marker.mass)
            * (marker.w * marker.w + tangential_speed_squared);
        shell.minimum_mass = std::min(shell.minimum_mass, marker.mass);
        shell.maximum_mass = std::max(shell.maximum_mass, marker.mass);
        const double vt = std::sqrt(2.0 * marker.eta)
            / std::max(marker.r, 1.0e-300);
        const double speed = std::hypot(marker.w, vt);
        shell.maximum_speed = std::max(shell.maximum_speed, speed);
    }
    shell.number_density = static_cast<double>(end - begin) / shell.volume;
    shell.density = static_cast<double>(shell_mass) / shell.volume;
    MeanFreePathEstimate mean_free_path;
    if (options.physics == WeightedPICCollisionPhysics::venus_cross_section
        && p.bp_cross_section == "constant"
        && shell.minimum_mass == shell.maximum_mass) {
        // The production SIDM case admits an exact one-pass result.  For a
        // constant cross section the relative speed cancels from the mean
        // free path, while the velocity dispersion follows from the moments
        // already accumulated for this shell.
        const long double mean_radial_velocity =
            radial_momentum / shell_mass;
        const long double centred_three_dimensional_variance = std::max(
            0.0L,
            speed_squared_moment / shell_mass
                - mean_radial_velocity * mean_radial_velocity);
        mean_free_path.orbital_velocity_dispersion = std::sqrt(
            static_cast<double>(
                centred_three_dimensional_variance / 3.0L));
        const double pair_cross_section = pair_total_cross_section(
            1.0,
            shell.minimum_mass,
            shell.minimum_mass,
            p,
            options);
        const double partner_number_density =
            static_cast<double>(end - begin - 1) / shell.volume;
        if (pair_cross_section > 0.0) {
            mean_free_path.mean_free_path =
                1.0 / (partner_number_density * pair_cross_section);
        }
        mean_free_path.pair_samples = static_cast<std::uint64_t>(
            (static_cast<long double>(end - begin)
             * static_cast<long double>(end - begin - 1))
            / 2.0L);
    } else {
        mean_free_path = velocity_averaged_mean_free_path(
            markers, order, begin, end, shell.volume, p, options);
    }
    shell.mean_free_path = mean_free_path.mean_free_path;
    shell.mean_free_path_pair_samples = mean_free_path.pair_samples;
    shell.orbital_velocity_dispersion =
        mean_free_path.orbital_velocity_dispersion;
    shell.majorant_u_sigma = pair_u_sigma_majorant(
        2.0 * shell.maximum_speed, shell.maximum_mass, p, options);
    return shell;
}

// Bisect an overly wide shell only at a genuine radial separation and with
// enough particles in both daughters. If population limits prevent further
// refinement, retain the shell and expose its locality ratio to the caller.
inline void build_shell_recursive(
    const std::vector<WeightedPICMarker>& markers,
    const std::vector<std::size_t>& order,
    const WeightedPICRadialMesh& mesh,
    double marker_mass,
    std::size_t begin,
    std::size_t end,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    std::vector<WeightedPICCollisionShell>& shells) {
    const WeightedPICCollisionShell shell = describe_shell(
        markers, order, mesh, marker_mass, begin, end, p, options);
    const std::size_t count = end - begin;
    const std::size_t middle = begin + count / 2;
    std::size_t split = end;
    const bool can_split_population =
        count >= 2 * options.minimum_shell_markers;
    if (can_split_population) {
        const std::size_t first_allowed =
            begin + options.minimum_shell_markers;
        const std::size_t last_allowed =
            end - options.minimum_shell_markers;
        for (std::size_t offset = 0;
             split == end && offset <= count;
             ++offset) {
            if (middle + offset >= first_allowed
                && middle + offset <= last_allowed
                && middle + offset > begin
                && middle + offset < end
                && markers[order[middle + offset - 1]].r
                    < markers[order[middle + offset]].r) {
                split = middle + offset;
                break;
            }
            if (offset > 0 && middle >= offset) {
                const std::size_t candidate = middle - offset;
                if (candidate >= first_allowed && candidate <= last_allowed
                    && candidate > begin && candidate < end
                    && markers[order[candidate - 1]].r
                        < markers[order[candidate]].r) {
                    split = candidate;
                }
            }
        }
    }
    const bool separable_radius = split != end;
    if (shell.outer_edge - shell.inner_edge > shell.mean_free_path
        && can_split_population
        && separable_radius) {
        build_shell_recursive(
            markers, order, mesh, marker_mass,
            begin, split, p, options, shells);
        build_shell_recursive(
            markers, order, mesh, marker_mass,
            split, end, p, options, shells);
        return;
    }
    shells.push_back(shell);
}

inline DSMCCollisionStats apply_shell(
    std::vector<WeightedPICMarker>& markers,
    const std::vector<std::size_t>& order,
    const WeightedPICCollisionShell& shell,
    double marker_mass,
    double dt,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    std::uint64_t orientation_seed,
    std::uint64_t event_seed,
    WeightedPICCollisionWorkspace::ThreadCache& cache,
    std::vector<WeightedPICCollisionEvent>* events = nullptr) {
    DSMCCollisionStats stats;
    const std::size_t count = shell.end - shell.begin;
    (void)marker_mass;
    if (count < 2 || !(dt > 0.0) || !(shell.number_density > 0.0)
        || p.bp_collision_strength == 0.0
        || !(shell.majorant_u_sigma > 0.0)) {
        return stats;
    }
    cache.velocity.resize(count);
    cache.initialized.assign(count, 0);
    cache.changed.assign(count, 0);
    cache.changed_local_indices.clear();
    // Reconstruct the unrecorded tangential azimuth deterministically when a
    // particle first participates. At the origin sample an isotropic direction for
    // its stored speed, preserving the spherical symmetry convention.
    auto velocity = [&](std::size_t local) -> DSMCVelocity& {
        if (cache.initialized[local] == 0) {
            const WeightedPICMarker& marker =
                markers[order[shell.begin + local]];
            const double vt = std::sqrt(2.0 * marker.eta)
                / std::max(marker.r, 1.0e-300);
            const double angle = 2.0 * PI * dsmc_splitmix_uniform_open_at(
                orientation_seed, marker.id);
            if (marker.r == 0.0) {
                const double cosine = 2.0 * dsmc_splitmix_uniform_open_at(
                    orientation_seed ^ UINT64_C(0x6f726967696e5f76), marker.id) - 1.0;
                const double sine = std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
                const double speed = std::abs(marker.w);
                cache.velocity[local] = {speed * cosine,
                    speed * sine * std::cos(angle), speed * sine * std::sin(angle)};
            } else {
                cache.velocity[local] = {
                    marker.w, vt * std::cos(angle), vt * std::sin(angle)};
            }
            cache.initialized[local] = 1;
        }
        return cache.velocity[local];
    };
    auto mark_changed = [&](std::size_t local) {
        if (cache.changed[local] == 0) {
            cache.changed[local] = 1;
            cache.changed_local_indices.push_back(local);
        }
    };

    stats.majorant_u_sigma = shell.majorant_u_sigma;
    double speed_bound = shell.maximum_speed;
    double clock = 0.0;
    double represented_event_mass = 0.0;
    DSMCSplitMix64 rng(event_seed);
    for (;;) {
        const double trial_rate = 0.5 * static_cast<double>(count - 1)
            * shell.number_density * stats.majorant_u_sigma;
        if (!(trial_rate > 0.0) || !std::isfinite(trial_rate)) {
            break;
        }
        clock += -std::log(rng.uniform_open()) / trial_rate;
        if (clock >= dt) {
            break;
        }
        ++stats.trial_pairs;
        const std::size_t ia = static_cast<std::size_t>(
            rng.uniform_index(static_cast<std::uint64_t>(count)));
        std::size_t ib = static_cast<std::size_t>(
            rng.uniform_index(static_cast<std::uint64_t>(count - 1)));
        if (ib >= ia) {
            ++ib;
        }
        if (options.at_most_one_collision_per_marker
            && (cache.changed[ia] != 0 || cache.changed[ib] != 0)) {
            continue;
        }
        const BPFAVec3 v1 = dsmc_to_vec3(velocity(ia));
        const BPFAVec3 v2 = dsmc_to_vec3(velocity(ib));
        const BPFAVec3 relative = bp_fa_sub(v1, v2);
        const double u = bp_fa_norm(relative);
        const double first_mass =
            markers[order[shell.begin + ia]].mass;
        const double second_mass =
            markers[order[shell.begin + ib]].mass;
        const double u_sigma = u
            * pair_total_cross_section(
                u, first_mass, second_mass, p, options);
        if (!(u_sigma > 0.0)) {
            continue;
        }
        if (u_sigma > stats.majorant_u_sigma
            * (1.0 + 64.0 * std::numeric_limits<double>::epsilon())) {
            throw std::runtime_error(
                "Weighted PIC collision majorant was violated.");
        }
        if (rng.uniform_open() * stats.majorant_u_sigma >= u_sigma) {
            continue;
        }

        const double total_mass = first_mass + second_mass;
        const BPFAVec3 momentum_before = bp_fa_add(
            bp_fa_scale(v1, first_mass),
            bp_fa_scale(v2, second_mass));
        const double energy_before =
            0.5 * (first_mass * bp_fa_dot(v1, v1)
                   + second_mass * bp_fa_dot(v2, v2));
        const BPFAVec3 centre =
            bp_fa_scale(momentum_before, 1.0 / total_mass);
        const double mu = sample_scatter_mu(
            u, first_mass, second_mass, p, options, rng);
        const double psi = 2.0 * PI * rng.uniform_open();
        BPFAVec3 direction;
        if (options.physics == WeightedPICCollisionPhysics::venus_cross_section
            && bp_fa_kernel_name(p) != "yukawa_born") {
            const double sint = std::sqrt(std::max(0.0, 1.0 - mu * mu));
            direction = {mu, sint * std::cos(psi), sint * std::sin(psi)};
        } else {
            direction = bp_fa_scattered_relative_direction(relative, mu, psi);
        }
        const BPFAVec3 relative_after = bp_fa_scale(direction, u);
        const BPFAVec3 out1 =
            bp_fa_add(
                centre,
                bp_fa_scale(relative_after, second_mass / total_mass));
        const BPFAVec3 out2 =
            bp_fa_sub(
                centre,
                bp_fa_scale(relative_after, first_mass / total_mass));
        cache.velocity[ia] = dsmc_from_vec3(out1);
        cache.velocity[ib] = dsmc_from_vec3(out2);
        if (events != nullptr) {
            events->push_back({
                order[shell.begin + ia],
                order[shell.begin + ib],
                v1,
                v2,
                out1,
                out2,
            });
        }
        mark_changed(ia);
        mark_changed(ib);
        ++stats.accepted_pairs;
        // The represented event mass is twice the reduced mass. For equal
        // masses it equals one marker mass; with a test-mass partner it tends
        // smoothly to zero.
        represented_event_mass +=
            2.0 * first_mass * second_mass / total_mass;

        const BPFAVec3 momentum_after = bp_fa_add(
            bp_fa_scale(out1, first_mass),
            bp_fa_scale(out2, second_mass));
        const double energy_after =
            0.5 * (first_mass * bp_fa_dot(out1, out1)
                   + second_mass * bp_fa_dot(out2, out2));
        stats.max_pair_momentum_error = std::max(
            stats.max_pair_momentum_error,
            bp_fa_norm(bp_fa_sub(momentum_after, momentum_before)));
        stats.max_pair_kinetic_energy_error = std::max(
            stats.max_pair_kinetic_energy_error,
            std::abs(energy_after - energy_before));
        const double new_speed = std::max(bp_fa_norm(out1), bp_fa_norm(out2));
        if (new_speed > speed_bound) {
            speed_bound = new_speed;
            stats.majorant_u_sigma = std::max(
                stats.majorant_u_sigma,
                pair_u_sigma_majorant(
                    2.0 * speed_bound, shell.maximum_mass, p, options));
        }
    }

    // Project only particles with accepted encounters back to spherical momentum.
    // Eta stores the final transverse kinetic energy at the unchanged radius, and
    // the independent budget compares before/after radial momentum and energy.
    for (const std::size_t local : cache.changed_local_indices) {
        WeightedPICMarker& marker = markers[order[shell.begin + local]];
        const double radius2 = std::max(
            marker.r * marker.r, std::numeric_limits<double>::min());
        const double kinetic_before = 0.5 * marker.mass
            * (marker.w * marker.w
               + 2.0 * marker.eta / radius2);
        const double momentum_before = marker.mass
            * static_cast<double>(mean_radial_velocity(marker));
        const DSMCVelocity& out = cache.velocity[local];
        marker.w = marker.r == 0.0
            ? std::hypot(out.x, std::hypot(out.y, out.z)) : out.x;
        marker.eta = 0.5 * marker.r * marker.r
            * (out.y * out.y + out.z * out.z);
        stats.momentum_error_x += marker.mass
            * static_cast<double>(mean_radial_velocity(marker)) - momentum_before;
        stats.kinetic_energy_error += 0.5 * marker.mass
            * (marker.w * marker.w
               + 2.0 * marker.eta / radius2)
            - kinetic_before;
    }
    stats.represented_event_density =
        represented_event_mass / shell.volume;
    return stats;
}

inline void accumulate(
    WeightedPICCollisionDiagnostics& target,
    const WeightedPICCollisionDiagnostics& source) {
    target.substeps += source.substeps;
    target.shell_visits += source.shell_visits;
    target.trial_pairs += source.trial_pairs;
    target.accepted_pairs += source.accepted_pairs;
    target.skipped_already_collided_trials +=
        source.skipped_already_collided_trials;
    target.active_markers = std::max(target.active_markers, source.active_markers);
    if (source.minimum_shell_population != 0) {
        target.minimum_shell_population = target.minimum_shell_population == 0
            ? source.minimum_shell_population
            : std::min(
                  target.minimum_shell_population,
                  source.minimum_shell_population);
    }
    target.nonlocal_shells += source.nonlocal_shells;
    target.nonlocal_occupied_shells += source.nonlocal_occupied_shells;
    target.minimum_mean_free_path = std::min(
        target.minimum_mean_free_path, source.minimum_mean_free_path);
    target.maximum_shell_width_over_mean_free_path = std::max(
        target.maximum_shell_width_over_mean_free_path,
        source.maximum_shell_width_over_mean_free_path);
    target.maximum_occupied_shell_span_over_mean_free_path = std::max(
        target.maximum_occupied_shell_span_over_mean_free_path,
        source.maximum_occupied_shell_span_over_mean_free_path);
    target.mean_free_path_pair_samples +=
        source.mean_free_path_pair_samples;
    target.minimum_orbital_velocity_dispersion = std::min(
        target.minimum_orbital_velocity_dispersion,
        source.minimum_orbital_velocity_dispersion);
    target.maximum_orbital_velocity_dispersion = std::max(
        target.maximum_orbital_velocity_dispersion,
        source.maximum_orbital_velocity_dispersion);
    target.maximum_relative_weight_spread = std::max(
        target.maximum_relative_weight_spread,
        source.maximum_relative_weight_spread);
    target.represented_event_mass += source.represented_event_mass;
    target.radial_momentum_error += source.radial_momentum_error;
    target.kinetic_energy_error += source.kinetic_energy_error;
    target.max_pair_momentum_error = std::max(
        target.max_pair_momentum_error, source.max_pair_momentum_error);
    target.max_pair_kinetic_energy_error = std::max(
        target.max_pair_kinetic_energy_error,
        source.max_pair_kinetic_energy_error);
}

inline std::uint64_t positive_radius_key(double radius) {
    std::uint64_t key = 0;
    if (radius == 0.0) return key;  // Both signed zeros have the same radius.
    std::memcpy(&key, &radius, sizeof(double));
    return key;
}

/// Stable parallel LSD radix ordering by radius from an ID-ordered input.
///
/// Active radii are finite and non-negative, so their IEEE-754 bit patterns
/// have the same order as the numerical values.  The input is already ordered
/// by marker ID, so eight stable radius passes reproduce the serial `(r,id)`
/// comparator exactly, including repeated radii.  Per-thread bucket
/// offsets are accumulated in lane order, which makes the result independent
/// of the OpenMP schedule.
// Stable byte passes order nonnegative finite radii exactly like a numeric
// sort. Starting from identity order resolves equal radii by persistent particle ID.
inline void parallel_radix_sort_active_markers(
    const std::vector<WeightedPICMarker>& markers,
    WeightedPICCollisionWorkspace& workspace,
    unsigned int requested_threads) {
    std::vector<std::size_t>& order = workspace.active_by_radius;
    const std::size_t count = order.size();
    if (count < 2) {
        return;
    }
    const int threads = std::max(
        1,
        std::min(
            static_cast<int>(requested_threads),
            static_cast<int>(std::min<std::size_t>(
                count,
                static_cast<std::size_t>(std::numeric_limits<int>::max())))));
#if !defined(EV_WEIGHTED_PIC_PARALLEL_RADIUS_SORT)
    (void)workspace;
    (void)threads;
    std::stable_sort(
        order.begin(), order.end(),
        [&](std::size_t a, std::size_t b) {
            if (markers[a].r != markers[b].r) {
                return markers[a].r < markers[b].r;
            }
            return markers[a].id < markers[b].id;
        });
#else
    if (threads == 1 || count < 4096) {
        std::stable_sort(
            order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) {
                if (markers[a].r != markers[b].r) {
                    return markers[a].r < markers[b].r;
                }
                return markers[a].id < markers[b].id;
            });
        return;
    }
    constexpr std::size_t radix = 256;
    constexpr int passes = 8;
    workspace.radix_scratch.resize(count);
    workspace.radix_histogram.assign(
        static_cast<std::size_t>(threads) * radix, 0);
    workspace.radix_offsets.resize(
        static_cast<std::size_t>(threads) * radix);
    workspace.radius_key_by_marker.resize(markers.size());
    parallel_for_chunks(
        0,
        static_cast<int>(count),
        threads,
        [&](int lo, int hi, int) {
            for (int ii = lo; ii < hi; ++ii) {
                const std::size_t marker_index =
                    order[static_cast<std::size_t>(ii)];
                workspace.radius_key_by_marker[marker_index] =
                    positive_radius_key(markers[marker_index].r);
            }
        });
    auto key = [&](std::size_t marker_index) {
        return workspace.radius_key_by_marker[marker_index];
    };
    std::vector<std::size_t>* source = &order;
    std::vector<std::size_t>* destination = &workspace.radix_scratch;
    for (int pass = 0; pass < passes; ++pass) {
        std::fill(
            workspace.radix_histogram.begin(),
            workspace.radix_histogram.end(),
            0);
        const int shift = 8 * pass;
        parallel_for_chunks(
            0,
            static_cast<int>(count),
            threads,
            [&](int lo, int hi, int tid) {
                std::uint64_t* histogram =
                    workspace.radix_histogram.data()
                    + static_cast<std::size_t>(tid) * radix;
                for (int ii = lo; ii < hi; ++ii) {
                    const std::uint64_t value = key(
                        (*source)[static_cast<std::size_t>(ii)]);
                    ++histogram[(value >> shift) & UINT64_C(0xff)];
                }
            });
        std::uint64_t bucket_start = 0;
        for (std::size_t digit = 0; digit < radix; ++digit) {
            std::uint64_t offset = bucket_start;
            for (int tid = 0; tid < threads; ++tid) {
                const std::size_t slot =
                    static_cast<std::size_t>(tid) * radix + digit;
                workspace.radix_offsets[slot] = offset;
                offset += workspace.radix_histogram[slot];
            }
            bucket_start = offset;
        }
        parallel_for_chunks(
            0,
            static_cast<int>(count),
            threads,
            [&](int lo, int hi, int tid) {
                std::uint64_t* offsets = workspace.radix_offsets.data()
                    + static_cast<std::size_t>(tid) * radix;
                for (int ii = lo; ii < hi; ++ii) {
                    const std::size_t marker_index =
                        (*source)[static_cast<std::size_t>(ii)];
                    const std::uint64_t value = key(marker_index);
                    const std::size_t digit = static_cast<std::size_t>(
                        (value >> shift) & UINT64_C(0xff));
                    (*destination)[offsets[digit]++] = marker_index;
                }
            });
        std::swap(source, destination);
    }
    if (source != &order) {
        throw std::runtime_error(
            "Weighted PIC radix ordering ended in the scratch buffer.");
    }
#endif
}

}  // namespace weighted_pic_collision_detail

inline const std::vector<WeightedPICCollisionShell>&
// Rebuild the active ordering and shell geometry from the current particles.
// A cached closed population is reused only after validating its identity order
// and boundary conditions; evolving radii still receive a fresh stable sort.
build_weighted_pic_collision_shells(
    const std::vector<WeightedPICMarker>& markers,
    const WeightedPICRadialMesh& mesh,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    WeightedPICCollisionWorkspace& workspace,
    double* marker_mass_out = nullptr,
    double* relative_weight_spread_out = nullptr,
    unsigned int requested_threads = 1) {
    weighted_pic_collision_detail::validate_options(options, p);
    if (mesh.edge.size() < 2 || mesh.shell_volume.size() + 1 != mesh.edge.size()) {
        throw std::runtime_error("Weighted PIC collision mesh is invalid.");
    }
    bool reuse_invariant_population =
        !options.input_ordered_by_radius
        && workspace.active_population_is_invariant
        && workspace.active_by_id.size() == markers.size()
        && p.inner_boundary == "reflecting" && using_reflecting_outer(p);
    if (reuse_invariant_population) {
        // A workspace may be reused with changed input or boundary controls.
        // Verify its cached population and secondary ID ordering before using
        // the stable radius sort. This retains the expensive sorting cache.
        const int validation_threads = std::max(1, static_cast<int>(
            std::min<std::size_t>(std::max(1U, requested_threads),
                std::min<std::size_t>(markers.size(),
                    static_cast<std::size_t>(std::numeric_limits<int>::max())))));
        std::vector<unsigned char> invariant(
            static_cast<std::size_t>(validation_threads), 1);
        parallel_for_chunks(0, static_cast<int>(markers.size()),
            static_cast<int>(invariant.size()), [&](int lo, int hi, int tid) {
                for (int ii = lo; ii < hi; ++ii) {
                    const std::size_t q = static_cast<std::size_t>(ii);
                    weighted_pic_collision_detail::validate_collision_marker(markers[q], mesh);
                    if (markers[q].state != WeightedPICMarkerState::active
                        || (q > 0 && markers[workspace.active_by_id[q]].id
                            < markers[workspace.active_by_id[q - 1]].id)) {
                        invariant[static_cast<std::size_t>(tid)] = 0;
                    }
                }
            });
        reuse_invariant_population = std::all_of(invariant.begin(), invariant.end(),
            [](unsigned char value) { return value != 0; });
    }
    if (reuse_invariant_population) {
        // Reflecting transport and elastic scattering preserve this exact ID
        // list.  Only radii and velocities evolve.
        workspace.active_by_radius = workspace.active_by_id;
    } else {
        const int collection_threads = std::max(
            1,
            std::min(
                static_cast<int>(requested_threads),
                static_cast<int>(std::min<std::size_t>(
                    markers.size(),
                    static_cast<std::size_t>(std::numeric_limits<int>::max())))));
        workspace.active_by_thread.resize(
            static_cast<std::size_t>(collection_threads));
        parallel_for_chunks(
            0,
            static_cast<int>(markers.size()),
            collection_threads,
            [&](int lo, int hi, int tid) {
                std::vector<std::size_t>& active =
                    workspace.active_by_thread[static_cast<std::size_t>(tid)];
                active.clear();
                active.reserve(static_cast<std::size_t>(hi - lo));
                for (int ii = lo; ii < hi; ++ii) {
                    const std::size_t i = static_cast<std::size_t>(ii);
                    weighted_pic_collision_detail::validate_collision_marker(markers[i], mesh);
                    if (markers[i].state == WeightedPICMarkerState::active) {
                        active.push_back(i);
                    }
                }
            });
        std::vector<std::size_t> active_offsets(
            static_cast<std::size_t>(collection_threads) + 1, 0);
        for (int tid = 0; tid < collection_threads; ++tid) {
            active_offsets[static_cast<std::size_t>(tid) + 1] =
                active_offsets[static_cast<std::size_t>(tid)]
                + workspace.active_by_thread[static_cast<std::size_t>(tid)].size();
        }
        workspace.active_by_radius.resize(active_offsets.back());
        parallel_for_chunks(
            0,
            collection_threads,
            collection_threads,
            [&](int lo, int hi, int) {
                for (int tid = lo; tid < hi; ++tid) {
                    const std::vector<std::size_t>& active =
                        workspace.active_by_thread[static_cast<std::size_t>(tid)];
                    std::copy(
                        active.begin(),
                        active.end(),
                        workspace.active_by_radius.begin()
                            + static_cast<std::ptrdiff_t>(
                                active_offsets[static_cast<std::size_t>(tid)]));
                }
            });
        workspace.active_population_is_invariant =
            workspace.active_by_radius.size() == markers.size()
            && p.inner_boundary == "reflecting"
            && using_reflecting_outer(p);
    }
    if (options.input_ordered_by_radius) {
        for (std::size_t q = 1; q < workspace.active_by_radius.size(); ++q) {
            const WeightedPICMarker& previous =
                markers[workspace.active_by_radius[q - 1]];
            const WeightedPICMarker& current =
                markers[workspace.active_by_radius[q]];
            if (current.r < previous.r
                || (current.r == previous.r && current.id < previous.id)) {
                throw std::runtime_error(
                    "Weighted PIC collision input claimed radial ordering but "
                    "violated the (r,id) order contract.");
            }
        }
    } else {
        if (!reuse_invariant_population) {
            workspace.active_by_id = workspace.active_by_radius;
            if (!std::is_sorted(
                    workspace.active_by_id.begin(),
                    workspace.active_by_id.end(),
                    [&](std::size_t a, std::size_t b) {
                        return markers[a].id < markers[b].id;
                    })) {
                std::stable_sort(
                    workspace.active_by_id.begin(),
                    workspace.active_by_id.end(),
                    [&](std::size_t a, std::size_t b) {
                        return markers[a].id < markers[b].id;
                    });
            }
            workspace.active_by_radius = workspace.active_by_id;
        }
        weighted_pic_collision_detail::parallel_radix_sort_active_markers(
            markers, workspace, requested_threads);
    }
    double spread = 0.0;
    const double marker_mass =
        weighted_pic_collision_detail::summarize_marker_masses(
            markers,
            workspace.active_by_radius,
            spread);
    if (marker_mass_out != nullptr) {
        *marker_mass_out = marker_mass;
    }
    if (relative_weight_spread_out != nullptr) {
        *relative_weight_spread_out = spread;
    }
    workspace.shells.clear();
    if (workspace.active_by_radius.empty()) {
        return workspace.shells;
    }
    const std::size_t count = workspace.active_by_radius.size();
    // For particles sampled at common radial centres, keep each radial tie group
    // together and assign its actual mesh shell volume. This representation has
    // different geometry from population-selected continuous-radius shells.
    if (options.radial_cell_shells) {
        for (std::size_t begin = 0; begin < count;) {
            std::size_t end = begin + 1;
            const double radius =
                markers[workspace.active_by_radius[begin]].r;
            while (end < count
                   && markers[workspace.active_by_radius[end]].r == radius) {
                ++end;
            }
            WeightedPICCollisionShell shell =
                weighted_pic_collision_detail::describe_shell(
                    markers,
                    workspace.active_by_radius,
                    mesh,
                    marker_mass,
                    begin,
                    end,
                    p,
                    options);
            const auto upper = std::upper_bound(
                mesh.edge.begin(), mesh.edge.end(), radius);
            const int radial_cell = std::clamp(
                static_cast<int>(upper - mesh.edge.begin()) - 1,
                0,
                static_cast<int>(mesh.edge.size()) - 2);
            shell.inner_edge = mesh.edge[static_cast<std::size_t>(radial_cell)];
            shell.outer_edge = mesh.edge[static_cast<std::size_t>(radial_cell + 1)];
            shell.volume = mesh.shell_volume[static_cast<std::size_t>(radial_cell)];
            double shell_mass = 0.0;
            shell.minimum_mass = std::numeric_limits<double>::infinity();
            shell.maximum_mass = 0.0;
            for (std::size_t q = begin; q < end; ++q) {
                const WeightedPICMarker& marker =
                    markers[workspace.active_by_radius[q]];
                shell_mass += marker.mass;
                shell.minimum_mass = std::min(
                    shell.minimum_mass, marker.mass);
                shell.maximum_mass = std::max(
                    shell.maximum_mass, marker.mass);
            }
            shell.number_density =
                static_cast<double>(end - begin) / shell.volume;
            shell.density = shell_mass / shell.volume;
            const weighted_pic_collision_detail::MeanFreePathEstimate
                mean_free_path =
                    weighted_pic_collision_detail::
                        velocity_averaged_mean_free_path(
                            markers,
                            workspace.active_by_radius,
                            begin,
                            end,
                            shell.volume,
                            p,
                            options);
            shell.mean_free_path = mean_free_path.mean_free_path;
            shell.mean_free_path_pair_samples = mean_free_path.pair_samples;
            shell.orbital_velocity_dispersion =
                mean_free_path.orbital_velocity_dispersion;
            shell.majorant_u_sigma =
                weighted_pic_collision_detail::pair_u_sigma_majorant(
                    2.0 * shell.maximum_speed,
                    shell.maximum_mass,
                    p,
                    options);
            workspace.shells.push_back(shell);
            begin = end;
        }
        return workspace.shells;
    }
    std::vector<std::pair<std::size_t, std::size_t>> blocks;
    blocks.reserve(
        (count + options.target_shell_markers - 1)
        / options.target_shell_markers);
    for (std::size_t begin = 0; begin < count;) {
        std::size_t end = std::min(
            count, begin + options.target_shell_markers);
        // A cell-centred Eulerian collision sample can contain many markers
        // at exactly one radius.  Keep that radial tie group in one shell so
        // a population target never creates coincident shell faces.
        while (end < count
               && markers[workspace.active_by_radius[end - 1]].r
                    == markers[workspace.active_by_radius[end]].r) {
            ++end;
        }
        if (count - end < options.minimum_shell_markers) {
            end = count;
        }
        blocks.emplace_back(begin, end);
        begin = end;
    }
    workspace.shells_by_block.clear();
    workspace.shells_by_block.resize(blocks.size());
    const int block_threads = std::max(
        1,
        std::min(
            static_cast<int>(requested_threads),
            static_cast<int>(blocks.size())));
    parallel_for_dynamic_chunks(
        0,
        static_cast<int>(blocks.size()),
        block_threads,
        1,
        [&](int lo, int hi, int) {
            for (int block_number = lo; block_number < hi; ++block_number) {
                const std::size_t block =
                    static_cast<std::size_t>(block_number);
                const auto [begin, end] = blocks[block];
                weighted_pic_collision_detail::build_shell_recursive(
                    markers,
                    workspace.active_by_radius,
                    mesh,
                    marker_mass,
                    begin,
                    end,
                    p,
                    options,
                    workspace.shells_by_block[block]);
            }
        });
    std::size_t shell_count = 0;
    for (const auto& block_shells : workspace.shells_by_block) {
        shell_count += block_shells.size();
    }
    workspace.shells.reserve(shell_count);
    for (const auto& block_shells : workspace.shells_by_block) {
        workspace.shells.insert(
            workspace.shells.end(), block_shells.begin(), block_shells.end());
    }
    return workspace.shells;
}

/// Refreshes the velocity-dependent data of one fixed collision shell.
/// During a split collision batch, elastic SIDM encounters change velocities
/// but not radii, marker masses, shell faces, density, or the constant-sigma
/// mean free path.  Recomputing only these kinematic quantities is therefore
/// sufficient before the shell takes its next local adaptive substep.
inline void refresh_weighted_pic_collision_shell_kinematics(
    const std::vector<WeightedPICMarker>& markers,
    const std::vector<std::size_t>& order,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    WeightedPICCollisionShell& shell) {
    long double mass = 0.0L;
    long double radial_momentum = 0.0L;
    long double speed_squared_moment = 0.0L;
    double maximum_speed = 0.0;
    for (std::size_t local = shell.begin; local < shell.end; ++local) {
        const WeightedPICMarker& marker = markers[order[local]];
        const long double tangential_speed_squared =
            weighted_pic_collision_detail::tangential_speed_squared(marker);
        mass += marker.mass;
        radial_momentum += static_cast<long double>(marker.mass)
            * weighted_pic_collision_detail::mean_radial_velocity(marker);
        speed_squared_moment += static_cast<long double>(marker.mass)
            * (marker.w * marker.w + tangential_speed_squared);
        maximum_speed = std::max(
            maximum_speed,
            std::hypot(
                marker.w,
                std::sqrt(static_cast<double>(tangential_speed_squared))));
    }
    const long double mean_radial_velocity = radial_momentum / mass;
    const long double centred_three_dimensional_variance = std::max(
        0.0L,
        speed_squared_moment / mass
            - mean_radial_velocity * mean_radial_velocity);
    shell.orbital_velocity_dispersion = std::sqrt(
        static_cast<double>(centred_three_dimensional_variance / 3.0L));
    shell.maximum_speed = maximum_speed;
    shell.majorant_u_sigma =
        weighted_pic_collision_detail::pair_u_sigma_majorant(
            2.0 * maximum_speed,
            shell.maximum_mass,
            p,
            options);
}

// With constant cross section and equal mass within each shell, elastic
// collisions keep radii, density and mean free path fixed. Only velocities and
// the proposal majorant need refreshing during the split interval.
inline bool weighted_pic_collision_shell_layout_is_reusable(
    const Params& p,
    const WeightedPICCollisionOptions& options,
    const WeightedPICCollisionWorkspace& workspace) {
    if (options.physics != WeightedPICCollisionPhysics::venus_cross_section
        || p.bp_cross_section != "constant"
        || workspace.shells.empty()) {
        return false;
    }
    return std::all_of(
        workspace.shells.begin(),
        workspace.shells.end(),
        [](const WeightedPICCollisionShell& shell) {
            return shell.minimum_mass == shell.maximum_mass;
        });
}

/// Refreshes the velocity moments and majorants after an equal-mass constant-
/// cross-section collision batch.  Collisions change velocities while
/// leaving every radius and mass fixed, so the stable radial order, shell
/// faces, and exact mean-free-path refinement remain valid.  Reusing that
/// geometry removes a second full marker sort at the same physical time.
inline void refresh_weighted_pic_collision_shell_kinematics(
    const std::vector<WeightedPICMarker>& markers,
    const WeightedPICRadialMesh& mesh,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    WeightedPICCollisionWorkspace& workspace,
    unsigned int requested_threads = 1) {
    if (!weighted_pic_collision_shell_layout_is_reusable(
            p, options, workspace)) {
        throw std::runtime_error(
            "Weighted PIC collision shell layout cannot be reused.");
    }
    (void)mesh;
    const int threads = std::max(
        1,
        std::min(
            static_cast<int>(requested_threads),
            static_cast<int>(workspace.shells.size())));
    parallel_for_dynamic_chunks(
        0,
        static_cast<int>(workspace.shells.size()),
        threads,
        1,
        [&](int lo, int hi, int) {
            for (int shell_number = lo; shell_number < hi; ++shell_number) {
                const std::size_t i =
                    static_cast<std::size_t>(shell_number);
                refresh_weighted_pic_collision_shell_kinematics(
                    markers,
                    workspace.active_by_radius,
                    p,
                    options,
                    workspace.shells[i]);
            }
        });
}

/// Conservative local collision time based on the NTC majorant.  The
/// accepted event rate is lower by the measured acceptance fraction, while
/// this bound guarantees that a collision scheduling interval never exceeds
/// the fastest shell rate represented by the NTC process.
// The inverse maximum finite-partner proposal rate supplies the strict
// local collision substep bound. It includes rejected NTC proposals.
inline double weighted_pic_collision_majorant_timescale(
    const std::vector<WeightedPICCollisionShell>& shells) {
    double result = std::numeric_limits<double>::infinity();
    for (const WeightedPICCollisionShell& shell : shells) {
        const std::size_t count = shell.end - shell.begin;
        const double partner_number_density = count > 1
            ? shell.number_density
                * static_cast<double>(count - 1)
                / static_cast<double>(count)
            : 0.0;
        const double rate =
            partner_number_density * shell.majorant_u_sigma;
        if (rate > 0.0) {
            result = std::min(result, 1.0 / rate);
        }
    }
    return result;
}

inline double weighted_pic_collision_timestep(
    const std::vector<WeightedPICCollisionShell>& shells,
    const WeightedPICCollisionOptions& options) {
    return options.collision_cfl
        * weighted_pic_collision_majorant_timescale(shells);
}

/// Estimates min_s [rho_s <u sigma/m>_s]^{-1} from a deterministic pilot
/// sample in every local shell.  The standard-error buffer limits the chance
/// that sampling noise makes the collision deadline too long.  The estimate
/// is capped by the exact NTC majorant, which remains the subcycling bound.
// A deterministic pilot estimates the accepted collision rate for a global
// operator schedule. Its statistical buffer is distinct from the strict majorant
// that continues to bound every local collision substep.
inline double weighted_pic_sampled_collision_timescale(
    const std::vector<WeightedPICMarker>& markers,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    const WeightedPICCollisionWorkspace& workspace,
    std::uint64_t seed = UINT64_C(0x70696374636f6c6c)) {
    double result = std::numeric_limits<double>::infinity();
    for (std::size_t shell_index = 0;
         shell_index < workspace.shells.size();
         ++shell_index) {
        const WeightedPICCollisionShell& shell = workspace.shells[shell_index];
        const std::size_t count = shell.end - shell.begin;
        if (count < 2 || !(shell.number_density > 0.0)
            || !(shell.majorant_u_sigma > 0.0)) {
            continue;
        }
        const std::size_t samples = options.scheduling_pair_samples_per_shell;
        DSMCSplitMix64 rng(
            seed ^ (static_cast<std::uint64_t>(shell.begin) << 1)
            ^ (static_cast<std::uint64_t>(shell.end) << 33)
            ^ static_cast<std::uint64_t>(shell_index));
        long double mean = 0.0L;
        long double m2 = 0.0L;
        std::size_t sampled_pairs = 0;
        std::size_t attempts = 0;
        const std::size_t maximum_attempts = 1024 * samples;
        while (sampled_pairs < samples && attempts++ < maximum_attempts) {
            const std::size_t first_local = static_cast<std::size_t>(
                rng.uniform_index(static_cast<std::uint64_t>(count)));
            std::size_t second_local = static_cast<std::size_t>(
                rng.uniform_index(static_cast<std::uint64_t>(count - 1)));
            if (second_local >= first_local) {
                ++second_local;
            }
            const WeightedPICMarker& first = markers[
                workspace.active_by_radius[shell.begin + first_local]];
            const WeightedPICMarker& second = markers[
                workspace.active_by_radius[shell.begin + second_local]];
            const double first_vt = std::sqrt(2.0 * first.eta)
                / std::max(first.r, 1.0e-300);
            const double second_vt = std::sqrt(2.0 * second.eta)
                / std::max(second.r, 1.0e-300);
            const double first_angle = 2.0 * PI * rng.uniform_open();
            const double second_angle = 2.0 * PI * rng.uniform_open();
            BPFAVec3 first_velocity{
                first.w,
                first_vt * std::cos(first_angle),
                first_vt * std::sin(first_angle),
            };
            BPFAVec3 second_velocity{
                second.w,
                second_vt * std::cos(second_angle),
                second_vt * std::sin(second_angle),
            };
            if (first.r == 0.0) first_velocity =
                weighted_pic_collision_detail::isotropic_origin_velocity(
                    first, 2.0 * rng.uniform_open() - 1.0, first_angle);
            if (second.r == 0.0) second_velocity =
                weighted_pic_collision_detail::isotropic_origin_velocity(
                    second, 2.0 * rng.uniform_open() - 1.0, second_angle);
            const double relative_speed = bp_fa_norm(
                bp_fa_sub(first_velocity, second_velocity));
            const double kernel = relative_speed
                * weighted_pic_collision_detail::pair_total_cross_section(
                    relative_speed,
                    first.mass,
                    second.mass,
                    p,
                    options);
            const long double delta = static_cast<long double>(kernel) - mean;
            ++sampled_pairs;
            mean += delta / static_cast<long double>(sampled_pairs);
            m2 += delta * (static_cast<long double>(kernel) - mean);
        }
        if (sampled_pairs == 0) {
            continue;
        }
        const long double variance = sampled_pairs > 1
            ? std::max(
                0.0L,
                m2 / static_cast<long double>(sampled_pairs - 1))
            : 0.0L;
        const long double standard_error = std::sqrt(
            variance / static_cast<long double>(sampled_pairs));
        const double buffered_mean = std::min(
            shell.majorant_u_sigma,
            static_cast<double>(
                mean + options.scheduling_standard_error_buffer
                    * standard_error));
        if (buffered_mean > 0.0) {
            const double partner_number_density =
                shell.number_density
                * static_cast<double>(count - 1)
                / static_cast<double>(count);
            result = std::min(
                result,
                1.0 / (partner_number_density * buffered_mean));
        }
    }
    return result;
}

inline WeightedPICCollisionDiagnostics apply_weighted_pic_collision_interval(
    std::vector<WeightedPICMarker>& markers,
    const WeightedPICRadialMesh& mesh,
    double dt,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    WeightedPICCollisionSamplerState& sampler,
    WeightedPICCollisionWorkspace& workspace,
    unsigned int requested_threads = 1,
    std::vector<WeightedPICCollisionEvent>* recorded_events = nullptr) {
    if (!(dt >= 0.0) || !std::isfinite(dt) || requested_threads == 0) {
        throw std::runtime_error("Invalid weighted PIC collision interval.");
    }
    weighted_pic_collision_detail::validate_options(options, p);
    WeightedPICCollisionDiagnostics total;
    if (dt == 0.0 || p.bp_collision_strength == 0.0) {
        return total;
    }
    // Disjoint shells can consume the same physical collision interval on
    // independent local clocks while radii remain frozen. This path requires the
    // constant-cross-section shell geometry to remain valid after every encounter.
    if (options.adaptive_shell_timestep) {
        // Marker radii do not move during this operator-split collision
        // interval, so disjoint radial shells may advance independently.
        // The optimized path is deliberately restricted to the case for
        // which shell geometry and mean free paths remain exact after every
        // local elastic update.  Other kernels retain the reference path.
        double marker_mass = 0.0;
        double weight_spread = 0.0;
        build_weighted_pic_collision_shells(
            markers,
            mesh,
            p,
            options,
            workspace,
            &marker_mass,
            &weight_spread,
            requested_threads);
        total.active_markers = workspace.active_by_radius.size();
        total.maximum_relative_weight_spread = weight_spread;
        if (workspace.shells.empty()) {
            return total;
        }
        if (!weighted_pic_collision_shell_layout_is_reusable(
                p, options, workspace)) {
            throw std::runtime_error(
                "Adaptive weighted PIC shell clocks currently require "
                "constant cross section and equal marker mass within every shell.");
        }

        for (const WeightedPICCollisionShell& shell : workspace.shells) {
            const std::size_t population = shell.end - shell.begin;
            total.minimum_shell_population =
                total.minimum_shell_population == 0
                ? population
                : std::min(total.minimum_shell_population, population);
            total.minimum_mean_free_path = std::min(
                total.minimum_mean_free_path, shell.mean_free_path);
            total.mean_free_path_pair_samples +=
                shell.mean_free_path_pair_samples;
            total.minimum_orbital_velocity_dispersion = std::min(
                total.minimum_orbital_velocity_dispersion,
                shell.orbital_velocity_dispersion);
            total.maximum_orbital_velocity_dispersion = std::max(
                total.maximum_orbital_velocity_dispersion,
                shell.orbital_velocity_dispersion);
            const double ratio = std::isfinite(shell.mean_free_path)
                    && shell.mean_free_path > 0.0
                ? (shell.outer_edge - shell.inner_edge) / shell.mean_free_path
                : 0.0;
            total.maximum_shell_width_over_mean_free_path = std::max(
                total.maximum_shell_width_over_mean_free_path, ratio);
            if (ratio > 1.0) {
                ++total.nonlocal_shells;
            }
            const double occupied_span =
                markers[workspace.active_by_radius[shell.end - 1]].r
                - markers[workspace.active_by_radius[shell.begin]].r;
            const double occupied_ratio = std::isfinite(shell.mean_free_path)
                    && shell.mean_free_path > 0.0
                ? occupied_span / shell.mean_free_path
                : 0.0;
            total.maximum_occupied_shell_span_over_mean_free_path = std::max(
                total.maximum_occupied_shell_span_over_mean_free_path,
                occupied_ratio);
            if (occupied_ratio > 1.0) {
                ++total.nonlocal_occupied_shells;
            }
        }
        if (options.fail_on_nonlocal_shell && total.nonlocal_shells != 0) {
            throw std::runtime_error(
                "Adaptive weighted PIC collision shell exceeds its mean free path; "
                "increase marker count or lower the minimum shell population.");
        }

        workspace.shell_stats.assign(
            workspace.shells.size(), DSMCCollisionStats{});
        workspace.shell_substeps.assign(workspace.shells.size(), 0);
        workspace.shell_events.assign(
            recorded_events != nullptr ? workspace.shells.size() : 0, {});
        const int threads = static_cast<int>(std::min<std::size_t>(
            requested_threads, workspace.shells.size()));
        workspace.thread_cache.resize(static_cast<std::size_t>(threads));
        const std::uint64_t epoch = sampler.next_counter++;
        parallel_for_chunks(
            0,
            static_cast<int>(workspace.shells.size()),
            threads,
            [&](int begin, int end, int tid) {
                for (int shell_number = begin;
                     shell_number < end;
                     ++shell_number) {
                    const std::size_t shell_index =
                        static_cast<std::size_t>(shell_number);
                    WeightedPICCollisionShell& shell =
                        workspace.shells[shell_index];
                    DSMCCollisionStats shell_total;
                    double elapsed = 0.0;
                    std::uint64_t local_substep = 0;
                    while (elapsed < dt) {
                        if (local_substep >= static_cast<std::uint64_t>(
                                options.maximum_substeps)) {
                            throw std::runtime_error(
                                "Adaptive weighted PIC collision shell exceeded "
                                "its local substep limit.");
                        }
                        const std::size_t count = shell.end - shell.begin;
                        const double partner_number_density = count > 1
                            ? shell.number_density
                                * static_cast<double>(count - 1)
                                / static_cast<double>(count)
                            : 0.0;
                        const double rate =
                            partner_number_density * shell.majorant_u_sigma;
                        const double remaining = dt - elapsed;
                        const double subdt = rate > 0.0
                            ? std::min(remaining, options.collision_cfl / rate)
                            : remaining;
                        if (!(subdt > 0.0) || !std::isfinite(subdt)) {
                            throw std::runtime_error(
                                "Adaptive weighted PIC shell timestep is invalid.");
                        }
                        const std::uint64_t stream =
                            shell_index
                            + workspace.shells.size() * local_substep;
                        const DSMCCollisionStats step =
                            weighted_pic_collision_detail::apply_shell(
                                markers,
                                workspace.active_by_radius,
                                shell,
                                marker_mass,
                                subdt,
                                p,
                                options,
                                weighted_pic_collision_detail::collision_seed(
                                    sampler.seed, epoch, UINT64_C(2) * stream),
                                weighted_pic_collision_detail::collision_seed(
                                    sampler.seed,
                                    epoch,
                                    UINT64_C(2) * stream + UINT64_C(1)),
                                workspace.thread_cache[
                                    static_cast<std::size_t>(tid)],
                                recorded_events != nullptr
                                    ? &workspace.shell_events[shell_index]
                                    : nullptr);
                        shell_total.trial_pairs += step.trial_pairs;
                        shell_total.accepted_pairs += step.accepted_pairs;
                        shell_total.majorant_u_sigma = std::max(
                            shell_total.majorant_u_sigma,
                            step.majorant_u_sigma);
                        shell_total.represented_event_density +=
                            step.represented_event_density;
                        shell_total.momentum_error_x += step.momentum_error_x;
                        shell_total.momentum_error_y += step.momentum_error_y;
                        shell_total.momentum_error_z += step.momentum_error_z;
                        shell_total.kinetic_energy_error +=
                            step.kinetic_energy_error;
                        shell_total.max_pair_momentum_error = std::max(
                            shell_total.max_pair_momentum_error,
                            step.max_pair_momentum_error);
                        shell_total.max_pair_kinetic_energy_error = std::max(
                            shell_total.max_pair_kinetic_energy_error,
                            step.max_pair_kinetic_energy_error);
                        elapsed = subdt >= remaining ? dt : elapsed + subdt;
                        ++local_substep;
                        if (elapsed < dt) {
                            refresh_weighted_pic_collision_shell_kinematics(
                                markers,
                                workspace.active_by_radius,
                                p,
                                options,
                                shell);
                        }
                    }
                    workspace.shell_stats[shell_index] = shell_total;
                    workspace.shell_substeps[shell_index] = local_substep;
                }
            });
        for (std::size_t i = 0; i < workspace.shells.size(); ++i) {
            const DSMCCollisionStats& stats = workspace.shell_stats[i];
            total.trial_pairs += stats.trial_pairs;
            total.accepted_pairs += stats.accepted_pairs;
            total.represented_event_mass +=
                stats.represented_event_density * workspace.shells[i].volume;
            total.radial_momentum_error += stats.momentum_error_x;
            total.kinetic_energy_error += stats.kinetic_energy_error;
            total.max_pair_momentum_error = std::max(
                total.max_pair_momentum_error,
                stats.max_pair_momentum_error);
            total.max_pair_kinetic_energy_error = std::max(
                total.max_pair_kinetic_energy_error,
                stats.max_pair_kinetic_energy_error);
            total.substeps = std::max(
                total.substeps, workspace.shell_substeps[i]);
            total.shell_visits += workspace.shell_substeps[i];
            if (recorded_events != nullptr) {
                recorded_events->insert(
                    recorded_events->end(),
                    workspace.shell_events[i].begin(),
                    workspace.shell_events[i].end());
            }
        }
        return total;
    }
    // The general reference path rebuilds shells and takes one common substep
    // limited by the fastest proposal rate. Every substep advances the saved random
    // epoch, and shell-order reduction records all accepted events.
    double elapsed = 0.0;
    while (elapsed < dt) {
        if (total.substeps >= static_cast<std::uint64_t>(options.maximum_substeps)) {
            throw std::runtime_error(
                "Weighted PIC collision interval exceeded its substep limit.");
        }
        double marker_mass = 0.0;
        double weight_spread = 0.0;
        build_weighted_pic_collision_shells(
            markers, mesh, p, options, workspace,
            &marker_mass, &weight_spread, requested_threads);
        total.active_markers = workspace.active_by_radius.size();
        total.maximum_relative_weight_spread = std::max(
            total.maximum_relative_weight_spread, weight_spread);
        if (workspace.shells.empty()) {
            break;
        }
        const double limit = weighted_pic_collision_timestep(
            workspace.shells, options);
        const double remaining = dt - elapsed;
        const double subdt = std::isfinite(limit)
            ? std::min(remaining, limit)
            : remaining;
        if (!(subdt > 0.0) || !std::isfinite(subdt)) {
            throw std::runtime_error(
                "Weighted PIC collision timestep is non-positive or non-finite.");
        }

        for (const WeightedPICCollisionShell& shell : workspace.shells) {
            const std::size_t population = shell.end - shell.begin;
            total.minimum_shell_population =
                total.minimum_shell_population == 0
                ? population
                : std::min(total.minimum_shell_population, population);
            total.minimum_mean_free_path = std::min(
                total.minimum_mean_free_path, shell.mean_free_path);
            total.mean_free_path_pair_samples +=
                shell.mean_free_path_pair_samples;
            total.minimum_orbital_velocity_dispersion = std::min(
                total.minimum_orbital_velocity_dispersion,
                shell.orbital_velocity_dispersion);
            total.maximum_orbital_velocity_dispersion = std::max(
                total.maximum_orbital_velocity_dispersion,
                shell.orbital_velocity_dispersion);
            const double ratio = std::isfinite(shell.mean_free_path)
                && shell.mean_free_path > 0.0
                ? (shell.outer_edge - shell.inner_edge) / shell.mean_free_path
                : 0.0;
            total.maximum_shell_width_over_mean_free_path = std::max(
                total.maximum_shell_width_over_mean_free_path, ratio);
            if (ratio > 1.0) {
                ++total.nonlocal_shells;
            }
            // The shell edges set the volume used for the density and mean
            // free path.  Empty gaps between neighboring shells, especially
            // the empty interval inside the innermost marker, do not measure
            // the separation of a pair that can actually collide.  Record
            // that physical separation independently, while retaining the
            // conservative edge-width diagnostic above.
            const double occupied_span =
                markers[workspace.active_by_radius[shell.end - 1]].r
                - markers[workspace.active_by_radius[shell.begin]].r;
            const double occupied_ratio =
                std::isfinite(shell.mean_free_path)
                    && shell.mean_free_path > 0.0
                ? occupied_span / shell.mean_free_path
                : 0.0;
            total.maximum_occupied_shell_span_over_mean_free_path = std::max(
                total.maximum_occupied_shell_span_over_mean_free_path,
                occupied_ratio);
            if (occupied_ratio > 1.0) {
                ++total.nonlocal_occupied_shells;
            }
        }
        if (options.fail_on_nonlocal_shell && total.nonlocal_shells != 0) {
            const WeightedPICCollisionShell* worst = nullptr;
            double worst_ratio = 0.0;
            for (const WeightedPICCollisionShell& shell : workspace.shells) {
                const double ratio = std::isfinite(shell.mean_free_path)
                    && shell.mean_free_path > 0.0
                    ? (shell.outer_edge - shell.inner_edge)
                        / shell.mean_free_path
                    : 0.0;
                if (ratio > worst_ratio) {
                    worst_ratio = ratio;
                    worst = &shell;
                }
            }
            std::ostringstream message;
            message
                << "Weighted PIC collision shell exceeds its mean free path"
                << "; maximum_width_over_mfp=" << worst_ratio;
            if (worst != nullptr) {
                message << ", radial_edges=[" << worst->inner_edge << ','
                        << worst->outer_edge << "]"
                        << ", population=" << (worst->end - worst->begin);
            }
            message
                << ". Increase marker count or lower the minimum shell population.";
            throw std::runtime_error(message.str());
        }

        workspace.shell_stats.assign(workspace.shells.size(), DSMCCollisionStats{});
        workspace.shell_events.assign(
            recorded_events != nullptr ? workspace.shells.size() : 0,
            {});
        if (workspace.shells.size()
            > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "Weighted PIC collision shell count exceeds the parallel-loop index range.");
        }
        const int threads = static_cast<int>(std::min<std::size_t>(
            requested_threads, workspace.shells.size()));
        workspace.thread_cache.resize(static_cast<std::size_t>(threads));
        const std::uint64_t counter = sampler.next_counter;
        parallel_for_chunks(
            0,
            static_cast<int>(workspace.shells.size()),
            threads,
            [&](int begin, int end, int tid) {
                    for (int shell_number = begin;
                         shell_number < end;
                         ++shell_number) {
                        const std::size_t shell_index =
                            static_cast<std::size_t>(shell_number);
                        workspace.shell_stats[shell_index] =
                            weighted_pic_collision_detail::apply_shell(
                                markers,
                                workspace.active_by_radius,
                                workspace.shells[shell_index],
                                marker_mass,
                                subdt,
                                p,
                                options,
                                weighted_pic_collision_detail::collision_seed(
                                    sampler.seed, counter,
                                    UINT64_C(2) * shell_index),
                                weighted_pic_collision_detail::collision_seed(
                                    sampler.seed, counter,
                                    UINT64_C(2) * shell_index + UINT64_C(1)),
                                workspace.thread_cache[
                                    static_cast<std::size_t>(tid)],
                                recorded_events != nullptr
                                    ? &workspace.shell_events[shell_index]
                                    : nullptr);
                    }
            });
        for (std::size_t i = 0; i < workspace.shells.size(); ++i) {
            const DSMCCollisionStats& stats = workspace.shell_stats[i];
            total.trial_pairs += stats.trial_pairs;
            total.accepted_pairs += stats.accepted_pairs;
            total.represented_event_mass +=
                stats.represented_event_density * workspace.shells[i].volume;
            total.radial_momentum_error += stats.momentum_error_x;
            total.kinetic_energy_error += stats.kinetic_energy_error;
            total.max_pair_momentum_error = std::max(
                total.max_pair_momentum_error,
                stats.max_pair_momentum_error);
            total.max_pair_kinetic_energy_error = std::max(
                total.max_pair_kinetic_energy_error,
                stats.max_pair_kinetic_energy_error);
            if (recorded_events != nullptr) {
                recorded_events->insert(
                    recorded_events->end(),
                    workspace.shell_events[i].begin(),
                    workspace.shell_events[i].end());
            }
        }
        ++total.substeps;
        total.shell_visits += workspace.shells.size();
        ++sampler.next_counter;
        elapsed += subdt;
        if (dt - elapsed <= 8.0 * std::numeric_limits<double>::epsilon() * dt) {
            elapsed = dt;
        }
    }
    return total;
}

template <class Transport>
inline WeightedPICCollisionDiagnostics
// Second-order operator splitting applies half the collision interval, the
// full transport interval and the remaining collision half. Both collision
// applications retain their own updated shell geometry and random stream.
advance_weighted_pic_strang_collisional_step(
    std::vector<WeightedPICMarker>& markers,
    const WeightedPICRadialMesh& mesh,
    double dt,
    const Params& p,
    const WeightedPICCollisionOptions& options,
    WeightedPICCollisionSamplerState& sampler,
    WeightedPICCollisionWorkspace& workspace,
    Transport&& transport,
    unsigned int requested_threads = 1) {
    WeightedPICCollisionDiagnostics result =
        apply_weighted_pic_collision_interval(
            markers, mesh, 0.5 * dt, p, options,
            sampler, workspace, requested_threads);
    transport(markers, dt);
    const WeightedPICCollisionDiagnostics second =
        apply_weighted_pic_collision_interval(
            markers, mesh, 0.5 * dt, p, options,
            sampler, workspace, requested_threads);
    weighted_pic_collision_detail::accumulate(result, second);
    return result;
}

}  // namespace ev
