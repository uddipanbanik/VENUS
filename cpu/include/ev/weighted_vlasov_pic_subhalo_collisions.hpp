#pragma once

// Unresolved subhalo encounters in spherical weighted particle evolution.
// Each particle is one complete Plummer-like physical object with evolving mass
// and internal energy. A single calibrated encounter law selects exclusive
// unbound-scattering and capture outcomes. Inputs are the particles, physical
// radial cells, elapsed time, common area normalization and random counter.
// Scattering rotates the relative velocity elastically; capture merges both
// objects and converts their relative orbital kinetic energy to internal energy.
// The returned budgets close represented mass and instantaneous pair momentum/
// kinetic-plus-internal energy. The subsequent field reconstruction measures the
// separate gravitational energy change caused by the remnant position and mass.

#include "ev/parallel.hpp"
#include "ev/weighted_vlasov_pic.hpp"
#include "ev/weighted_vlasov_pic_subhalo_kernel_table.hpp"

#include <array>
#include <iomanip>
#include <sstream>

namespace ev {

enum class WeightedPICSubhaloPairSampling {
    maximum_weight_rejection,
    maximum_weight_importance,
};

struct WeightedPICSubhaloCollisionOptions {
    // Used only to initialize a fresh marker population.  Encounter kernels
    // use each marker's evolving internal_velocity_dispersion.
    double velocity_scale = 1.0;
    // Global Pi*a^2/M scale.  Scattering and capture must share this factor:
    // their relative normalization is fixed by the Plummer encounter table.
    double plummer_area_per_mass = 1.0;
    // The calibrated kernel begins at x=u/sigma_pair=0.05.  Speeds below the
    // chosen floor use the cross sections at the floor, preventing an
    // unresolved focusing divergence.  The floor therefore cannot lie below
    // the first calibrated speed.
    double low_speed_regularization = 0.05;
    double strength = 0.0;
    double collision_cfl = 0.2;
    int maximum_substeps = 256;
    // Advance every disjoint radial collision cell with its own local CFL
    // clock inside one operator batch.  The reference mode instead applies
    // the fastest cell's timestep to the complete mesh.
    bool adaptive_cell_timestep = false;
    // Virtual equal-volume collision shells covering
    // 0 <= r < mesh.edge.front().  One shell reproduces the Poisson solver's
    // single uniform hidden-core control volume.  More shells reduce the
    // geometric width of the widest (central) shell as
    // r_min / cbrt(inner_collision_cells), without implying that Monte-Carlo
    // lanes are spatial cells.  This does not guarantee a smaller
    // Delta-r/lambda_mfp: a velocity-dependent capture area can grow as the
    // central orbital speed falls, shortening lambda_mfp faster than the shell
    // narrows.  Any override therefore requires an evolving-DF locality test
    // and must be frozen in the run contract.
    int inner_collision_cells = 1;
    // Independent Monte-Carlo lanes inside every physical collision cell.  A
    // lane owns 1/N of the same physical cell volume and markers are
    // reassigned by a deterministic epoch-dependent hash at every collision
    // interval.  Summing the N within-lane NTC rates is an unbiased estimator
    // of the original all-pairs rate and exposes dense cusp cells to multiple
    // worker threads.  This is computational stratification, not extra radial
    // resolution.
    int collision_lanes = 1;
    // Both supported pair-sampling names select pair-majorant importance
    // sampling. Acceptance divides by the proposal pair bound, preserving
    // the physical rate for unequal marker weights.
    WeightedPICSubhaloPairSampling pair_sampling =
        WeightedPICSubhaloPairSampling::maximum_weight_rejection;
    std::size_t minimum_cell_population = 2;
    bool enable_scattering = true;
    bool enable_capture = true;
};

struct WeightedPICSubhaloCollisionDiagnostics {
    std::uint64_t substeps = 0;
    std::uint64_t cell_visits = 0;
    std::uint64_t trial_pairs = 0;
    std::uint64_t accepted_pairs = 0;
    std::uint64_t scattering_events = 0;
    std::uint64_t capture_events = 0;
    std::uint64_t inner_cell_visits = 0;
    std::uint64_t inner_scattering_events = 0;
    std::uint64_t inner_capture_events = 0;
    std::uint64_t skipped_already_collided_trials = 0;
    std::size_t active_markers_before = 0;
    std::size_t active_markers_after = 0;
    std::size_t spawned_markers = 0;
    std::size_t removed_markers = 0;
    std::size_t nonlocal_cells = 0;
    double represented_scattering_mass = 0.0;
    double represented_capture_mass = 0.0;
    double orbital_kinetic_energy_change = 0.0;
    double internal_energy_change = 0.0;
    double maximum_pair_momentum_error = 0.0;
    double maximum_pair_total_energy_error = 0.0;
    double minimum_mean_free_path = std::numeric_limits<double>::infinity();
    // Shortest local waiting time for one physical encounter,
    // t_coll = 1/[n <u Sigma_tot(u)>], measured from the evolving pair sample.
    double minimum_collision_timescale =
        std::numeric_limits<double>::infinity();
    double maximum_cell_width_over_mean_free_path = 0.0;
    std::uint64_t mean_free_path_pair_samples = 0;
    double minimum_orbital_velocity_dispersion =
        std::numeric_limits<double>::infinity();
    double maximum_orbital_velocity_dispersion = 0.0;
    double maximum_weight_ratio = 1.0;
    double minimum_internal_velocity_dispersion =
        std::numeric_limits<double>::infinity();
    double maximum_internal_velocity_dispersion = 0.0;
    double minimum_pair_velocity_scale =
        std::numeric_limits<double>::infinity();
    double maximum_pair_velocity_scale = 0.0;
};

struct WeightedPICSubhaloCollisionWorkspace {
    std::vector<std::vector<std::size_t>> cell_members;
    std::vector<std::vector<std::size_t>> physical_cell_members;
    std::vector<std::vector<std::size_t>> cell_pair_order;
    std::vector<std::vector<long double>> cell_pair_cumulative_weight;
    std::vector<std::vector<long double>> cell_pair_prefix_root_mass;
    std::vector<std::vector<long double>> cell_pair_prefix_mass;
    // One generation tag per local proposal position.  The collision loop
    // advances the generation at every adaptive substep, so testing a tag is
    // equivalent to clearing a byte-valued `collided` array.  Keeping these
    // buffers in the reusable workspace avoids allocating and zero-filling
    // every marker-sized array at every substep; this is particularly
    // important in dense high-gamma cusps where one collision interval can
    // require millions of substeps.  Tags are reset once per operator call,
    // and therefore cannot change the accepted-pair sequence or physics.
    std::vector<std::vector<std::uint32_t>> cell_collided_generation;
    std::vector<WeightedPICSubhaloCollisionDiagnostics> cell_diagnostics;
    std::vector<WeightedPICSubhaloCollisionDiagnostics>
        physical_cell_diagnostics;
    std::vector<unsigned char> remove_marker;
};


namespace weighted_pic_subhalo_unequal_plummer_detail {

using weighted_pic_subhalo_plummer_table::capture;
using weighted_pic_subhalo_plummer_table::mass_ratio;
using weighted_pic_subhalo_plummer_table::scatter;
using weighted_pic_subhalo_plummer_table::scatter_high_speed_coefficient;
using weighted_pic_subhalo_plummer_table::speed;
using weighted_pic_subhalo_plummer_table::speed_count;

struct MassRatioBracket {
    std::size_t low = 0;
    std::size_t high = 0;
    double fraction = 0.0;
};

/// Symmetric physical mass ratio q=min(M1,M2)/max(M1,M2).
inline double pair_mass_ratio(double first_mass, double second_mass) {
    if (!(first_mass > 0.0) || !(second_mass > 0.0)
        || !std::isfinite(first_mass) || !std::isfinite(second_mass)) {
        throw std::runtime_error(
            "Subhalo encounter kernel requires two positive masses.");
    }
    return std::min(first_mass, second_mass)
        / std::max(first_mass, second_mass);
}

/// Locate q on the logarithmic calibration grid.  Ratios below the smallest
/// calibrated value use its converged extreme-mass-ratio row.
inline MassRatioBracket mass_ratio_bracket(double value) {
    if (value <= mass_ratio.front()) {
        return {0, 0, 0.0};
    }
    if (value >= mass_ratio.back()) {
        const std::size_t last = mass_ratio.size() - 1;
        return {last, last, 0.0};
    }
    const auto upper = std::lower_bound(
        mass_ratio.begin(), mass_ratio.end(), value);
    const std::size_t high =
        static_cast<std::size_t>(upper - mass_ratio.begin());
    if (*upper == value) {
        return {high, high, 0.0};
    }
    const std::size_t low = high - 1;
    const double fraction =
        (std::log(value) - std::log(mass_ratio[low]))
        / (std::log(mass_ratio[high]) - std::log(mass_ratio[low]));
    return {low, high, fraction};
}

/// Interpolate one calibrated row in log speed.  Positive cross sections use
/// geometric interpolation; a zero endpoint uses a non-negative linear
/// transition so a channel can close without taking log(0).
// Interpolate positive calibrated areas geometrically in log speed. A
// channel with a zero endpoint uses a nonnegative linear transition, retaining
// its finite closure rather than evaluating log(0).
inline double interpolate_speed(
    double value,
    const std::array<double, speed_count>& row) {
    if (value <= speed.front()) {
        return row.front();
    }
    if (value >= speed.back()) {
        return row.back();
    }
    const auto upper = std::upper_bound(
        speed.begin(), speed.end(), value);
    const std::size_t high =
        static_cast<std::size_t>(upper - speed.begin());
    const std::size_t low = high - 1;
    const double fraction =
        (std::log(value) - std::log(speed[low]))
        / (std::log(speed[high]) - std::log(speed[low]));
    if (row[low] > 0.0 && row[high] > 0.0) {
        return std::exp(
            (1.0 - fraction) * std::log(row[low])
            + fraction * std::log(row[high]));
    }
    return std::max(
        0.0, (1.0 - fraction) * row[low] + fraction * row[high]);
}

/// Interpolate in log mass ratio.  The geometric rule is used when both
/// neighboring cross sections are positive; otherwise the linear rule
/// preserves a continuously closing capture channel.
inline double interpolate_mass_ratio(
    double first,
    double second,
    double fraction) {
    if (first > 0.0 && second > 0.0) {
        return std::exp(
            (1.0 - fraction) * std::log(first)
            + fraction * std::log(second));
    }
    return std::max(
        0.0, (1.0 - fraction) * first + fraction * second);
}

inline double capture_shape(
    double value,
    double pair_mass_ratio_value,
    double regularization) {
    const double regulated = std::max(value, regularization);
    if (regulated >= speed.back()) {
        return 0.0;
    }
    const MassRatioBracket bracket =
        mass_ratio_bracket(pair_mass_ratio_value);
    const double low = interpolate_speed(
        regulated, capture[bracket.low]);
    if (bracket.low == bracket.high) {
        return low;
    }
    const double high = interpolate_speed(
        regulated, capture[bracket.high]);
    return interpolate_mass_ratio(
        low, high, bracket.fraction);
}

inline double scatter_shape(
    double value,
    double pair_mass_ratio_value,
    double regularization) {
    const double regulated = std::max(value, regularization);
    const MassRatioBracket bracket =
        mass_ratio_bracket(pair_mass_ratio_value);
    if (regulated >= speed.back()) {
        const double low =
            scatter_high_speed_coefficient[bracket.low];
        const double coefficient = bracket.low == bracket.high
            ? low
            : interpolate_mass_ratio(
                low,
                scatter_high_speed_coefficient[bracket.high],
                bracket.fraction);
        const double x2 = regulated * regulated;
        return coefficient / (x2 * x2);
    }
    const double low = interpolate_speed(
        regulated, scatter[bracket.low]);
    if (bracket.low == bracket.high) {
        return low;
    }
    const double high = interpolate_speed(
        regulated, scatter[bracket.high]);
    return interpolate_mass_ratio(
        low, high, bracket.fraction);
}

}  // namespace weighted_pic_subhalo_unequal_plummer_detail

inline double weighted_pic_subhalo_scattering_cross_section_per_mass(
    double relative_speed,
    double velocity_scale,
    const WeightedPICSubhaloCollisionOptions& options) {
    if (!options.enable_scattering) {
        return 0.0;
    }
    const double x = relative_speed / velocity_scale;
    return options.strength * options.plummer_area_per_mass
        * weighted_pic_subhalo_unequal_plummer_detail::scatter_shape(
            x, 1.0, options.low_speed_regularization);
}

inline double weighted_pic_subhalo_scattering_cross_section_per_mass(
    double relative_speed,
    const WeightedPICSubhaloCollisionOptions& options) {
    return weighted_pic_subhalo_scattering_cross_section_per_mass(
        relative_speed, options.velocity_scale, options);
}

inline double weighted_pic_subhalo_capture_cross_section_per_mass(
    double relative_speed,
    double velocity_scale,
    const WeightedPICSubhaloCollisionOptions& options) {
    if (!options.enable_capture) {
        return 0.0;
    }
    const double x = relative_speed / velocity_scale;
    return options.strength * options.plummer_area_per_mass
        * weighted_pic_subhalo_unequal_plummer_detail::capture_shape(
            x, 1.0, options.low_speed_regularization);
}

/// Pair area scale for two homologous Plummer objects.
///
/// A=pi*a^2/M is configured by plummer_area_per_mass, so
/// a_i=sqrt(A M_i/pi).  The equal-mass table uses pi*a^2; replacing a by the
/// mean pair radius (a_i+a_j)/2 gives
/// A_ij=A(sqrt(M_i)+sqrt(M_j))^2/4.  In general the tabulated cross sections
/// scale as A_ij times a function of u/sigma_ij.  Only while the stored
/// dispersions obey the homologous virial relation sigma_i^2 proportional to
/// sqrt(M_i) do the high-speed tails reduce to the familiar
/// (M_i+M_j)^2/u^4 Rutherford and
/// (M_i+M_j)(a_i+a_j)/u^2 focusing scalings.  Capture heating evolves the
/// dispersions independently, so callers must use the full pair law below.
// Homologous radius a_i=sqrt(A*M_i/pi) gives pair area
// A_ij=A*(sqrt(M_i)+sqrt(M_j))^2/4. Use the evolving internal dispersion in
// the dimensionless speed; capture heating can change its initial mass scaling.
inline double weighted_pic_subhalo_pair_area(
    double first_mass,
    double second_mass,
    const WeightedPICSubhaloCollisionOptions& options) {
    if (!(first_mass > 0.0) || !(second_mass > 0.0)
        || !std::isfinite(first_mass) || !std::isfinite(second_mass)) {
        throw std::runtime_error(
            "Subhalo pair cross section requires two positive masses.");
    }
    const double root_sum =
        std::sqrt(first_mass) + std::sqrt(second_mass);
    return 0.25 * options.plummer_area_per_mass
        * root_sum * root_sum;
}

inline double weighted_pic_subhalo_scattering_cross_section(
    double relative_speed,
    double first_mass,
    double second_mass,
    double velocity_scale,
    const WeightedPICSubhaloCollisionOptions& options) {
    if (!options.enable_scattering) {
        return 0.0;
    }
    const double x = relative_speed / velocity_scale;
    const double mass_ratio =
        weighted_pic_subhalo_unequal_plummer_detail::pair_mass_ratio(
            first_mass, second_mass);
    return options.strength
        * weighted_pic_subhalo_pair_area(first_mass, second_mass, options)
        * weighted_pic_subhalo_unequal_plummer_detail::scatter_shape(
            x, mass_ratio, options.low_speed_regularization);
}

inline double weighted_pic_subhalo_capture_cross_section(
    double relative_speed,
    double first_mass,
    double second_mass,
    double velocity_scale,
    const WeightedPICSubhaloCollisionOptions& options) {
    if (!options.enable_capture) {
        return 0.0;
    }
    const double x = relative_speed / velocity_scale;
    const double mass_ratio =
        weighted_pic_subhalo_unequal_plummer_detail::pair_mass_ratio(
            first_mass, second_mass);
    return options.strength
        * weighted_pic_subhalo_pair_area(first_mass, second_mass, options)
        * weighted_pic_subhalo_unequal_plummer_detail::capture_shape(
            x, mass_ratio, options.low_speed_regularization);
}

inline double weighted_pic_subhalo_capture_cross_section_per_mass(
    double relative_speed,
    const WeightedPICSubhaloCollisionOptions& options) {
    return weighted_pic_subhalo_capture_cross_section_per_mass(
        relative_speed, options.velocity_scale, options);
}

inline double weighted_pic_subhalo_u_sigma_majorant(
    double velocity_scale,
    const WeightedPICSubhaloCollisionOptions& options) {
    const double dimensionless =
        (options.enable_capture
             ? weighted_pic_subhalo_plummer_table::capture_u_shape_bound
             : 0.0)
        + (options.enable_scattering
               ? weighted_pic_subhalo_plummer_table::scatter_u_shape_bound
               : 0.0);
    return options.strength * options.plummer_area_per_mass
        * velocity_scale * dimensionless;
}

/// Safe physical kernel bound for one complete unequal-mass pair.
///
/// Writing x=u/sigma_ij, the tabulated kernel is
/// u*Sigma_ij = strength*A_ij*sigma_ij*x*shape(x).  The pair velocity scale is
/// a positive weighted RMS of the two internal dispersions and therefore does
/// not exceed max(sigma_i,sigma_j).  Replacing it by that maximum gives a
/// rigorous pair-specific rejection bound without applying the largest marker
/// mass and dispersion to every other pair in the cell.
inline double weighted_pic_subhalo_pair_u_sigma_majorant(
    double first_mass,
    double second_mass,
    double first_internal_dispersion,
    double second_internal_dispersion,
    const WeightedPICSubhaloCollisionOptions& options) {
    if (!(first_internal_dispersion > 0.0)
        || !(second_internal_dispersion > 0.0)
        || !std::isfinite(first_internal_dispersion)
        || !std::isfinite(second_internal_dispersion)) {
        throw std::runtime_error(
            "Subhalo pair majorant requires two positive internal dispersions.");
    }
    const double maximum_dispersion =
        std::max(first_internal_dispersion, second_internal_dispersion);
    const double equal_reference_mass_area =
        options.plummer_area_per_mass * maximum_dispersion;
    const double dimensionless_kernel =
        weighted_pic_subhalo_u_sigma_majorant(
            maximum_dispersion, options)
        / equal_reference_mass_area;
    return weighted_pic_subhalo_pair_area(
               first_mass, second_mass, options)
        * maximum_dispersion * dimensionless_kernel
        * (1.0 + 128.0 * std::numeric_limits<double>::epsilon());
}

inline double weighted_pic_subhalo_u_sigma_majorant(
    const WeightedPICSubhaloCollisionOptions& options) {
    return weighted_pic_subhalo_u_sigma_majorant(
        options.velocity_scale, options);
}

/// Frank-compatible mean free path for a velocity-dependent pair kernel.
///
/// The trial pairs are uniform samples of the local evolving distribution
/// function.  If u is their relative speed and Sigma_ij(u) is the complete
/// unequal-mass pair area, then
///
///   lambda_mfp = <u> / [n_partner <u Sigma_ij(u)>],
///
/// where n_partner=(N-1)/V is the number density of other objects available
/// to one marker in a finite collision cell.  Replacing n_partner by N/V is
/// the large-N continuum approximation; retaining N-1 keeps the diagnostic
/// consistent with the unordered-pair collision rate even in sparse cells.
///
/// For equal masses and a velocity-independent Sigma=m*(sigma/m), this
/// approaches Frank & Tokayer's lambda_mfp=1/[rho*(sigma/m)] as N grows.
/// Keeping the two sums rather than separately rounded means also makes this
/// helper suitable for combining independent Monte-Carlo lanes.
inline double weighted_pic_subhalo_mean_free_path_from_pair_samples(
    double partner_number_density,
    long double relative_speed_sum,
    long double collision_kernel_sum) {
    if (!(partner_number_density > 0.0)
        || !(relative_speed_sum > 0.0L)
        || !(collision_kernel_sum > 0.0L)) {
        return std::numeric_limits<double>::infinity();
    }
    const long double inverse_length =
        static_cast<long double>(partner_number_density)
        * collision_kernel_sum / relative_speed_sum;
    if (!(inverse_length > 0.0L)
        || !std::isfinite(static_cast<double>(inverse_length))) {
        return std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(1.0L / inverse_length);
}

namespace weighted_pic_subhalo_collision_detail {

using Vec3 = std::array<double, 3>;

inline Vec3 add(const Vec3& a, const Vec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline Vec3 subtract(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline Vec3 scale(const Vec3& a, double factor) {
    return {factor * a[0], factor * a[1], factor * a[2]};
}

inline double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double norm(const Vec3& value) {
    return std::sqrt(std::max(0.0, dot(value, value)));
}

inline double marker_internal_velocity_dispersion(
    const WeightedPICMarker& marker) {
    return marker.internal_velocity_dispersion;
}

// Weight the two internal dispersions by their homologous radius factors
// sqrt(M). This positive RMS is bounded by the larger dispersion, allowing a
// finite pair-specific proposal majorant.
inline double pair_velocity_scale(
    const WeightedPICMarker& first,
    const WeightedPICMarker& second) {
    const double first_sigma =
        marker_internal_velocity_dispersion(first);
    const double second_sigma =
        marker_internal_velocity_dispersion(second);
    const double first_radius_weight = std::sqrt(first.mass);
    const double second_radius_weight = std::sqrt(second.mass);
    return std::sqrt(
        (first_radius_weight * first_sigma * first_sigma
         + second_radius_weight * second_sigma * second_sigma)
        / (first_radius_weight + second_radius_weight));
}

// The stored absolute internal energy must equal (3/2)*M*sigma_internal^2.
// Check that relation before using the dispersion to set an encounter area.
inline void enforce_internal_state(
    const WeightedPICMarker& marker,
    const char* context) {
    if (!(marker.internal_velocity_dispersion > 0.0)
        || !std::isfinite(marker.internal_velocity_dispersion)) {
        throw std::runtime_error(
            std::string(context)
            + " requires a positive marker internal velocity dispersion.");
    }
    const double expected = 1.5 * marker.mass
        * marker.internal_velocity_dispersion
        * marker.internal_velocity_dispersion;
    const double scale = std::max({
        1.0e-300, std::abs(expected), std::abs(marker.internal_energy)});
    if (std::abs(marker.internal_energy - expected)
        > 2048.0 * std::numeric_limits<double>::epsilon() * scale) {
        std::ostringstream message;
        message << std::setprecision(17)
                << context
                << " found inconsistent marker internal state: E_int="
                << marker.internal_energy << ", expected=" << expected
                << ", mass=" << marker.mass << ", sigma_s="
                << marker.internal_velocity_dispersion << '.';
        throw std::runtime_error(message.str());
    }
}

inline void update_internal_velocity_dispersion(
    WeightedPICMarker& marker) {
    marker.internal_velocity_dispersion = std::sqrt(
        std::max(
            0.0, 2.0 * marker.internal_energy / (3.0 * marker.mass)));
}

inline void validate_options(
    const WeightedPICSubhaloCollisionOptions& options) {
    if (!(options.velocity_scale > 0.0)
        || !std::isfinite(options.velocity_scale)
        || !(options.plummer_area_per_mass > 0.0)
        || !std::isfinite(options.plummer_area_per_mass)
        || !(options.low_speed_regularization
             >= weighted_pic_subhalo_plummer_table::speed.front())
        || !(options.low_speed_regularization < 1.0)
        || !std::isfinite(options.low_speed_regularization)
        || !(options.strength >= 0.0) || !std::isfinite(options.strength)
        || !(options.collision_cfl > 0.0)
        || !std::isfinite(options.collision_cfl)
        || options.maximum_substeps < 1
        || options.inner_collision_cells < 1
        || options.inner_collision_cells > 256
        || options.collision_lanes < 1
        || options.collision_lanes > 256
        || options.minimum_cell_population < 2) {
        throw std::runtime_error(
            "Invalid weighted-PIC subhalo collision options.");
    }
    if (options.strength > 0.0
        && !options.enable_scattering && !options.enable_capture) {
        throw std::runtime_error(
            "Enabled subhalo collisions require a nonzero channel.");
    }
}

inline std::size_t resolved_radial_cell(
    double radius,
    const WeightedPICRadialMesh& mesh) {
    if (radius < mesh.edge.front() || radius > mesh.edge.back()) {
        throw std::runtime_error(
            "Active subhalo-collision marker lies outside the radial mesh.");
    }
    const auto upper = std::upper_bound(
        mesh.edge.begin(), mesh.edge.end(), radius);
    return std::min(
        static_cast<std::size_t>(upper - mesh.edge.begin() - 1),
        mesh.shell_volume.size() - 1);
}

inline std::size_t collision_cell_count(
    const WeightedPICRadialMesh& mesh,
    const WeightedPICSubhaloCollisionOptions& options) {
    return (static_cast<std::size_t>(options.inner_collision_cells)
            + mesh.shell_volume.size())
        * static_cast<std::size_t>(options.collision_lanes);
}

// Physical radial cells include equal-volume shells inside the field face.
// An epoch-dependent identity hash assigns computational lanes within each cell;
// those lanes subdivide sampling volume and retain the physical radial width.
inline std::size_t collision_cell(
    double radius,
    std::uint64_t marker_id,
    std::uint64_t epoch,
    const WeightedPICRadialMesh& mesh,
    const WeightedPICSubhaloCollisionOptions& options) {
    if (!(radius >= 0.0) || radius > mesh.edge.back()
        || !std::isfinite(radius)) {
        throw std::runtime_error(
            "Active subhalo-collision marker lies outside the collision domain.");
    }
    const std::size_t inner =
        static_cast<std::size_t>(options.inner_collision_cells);
    const std::size_t lanes =
        static_cast<std::size_t>(options.collision_lanes);
    const std::uint64_t lane_hash = weighted_pic_mix64(
        marker_id
        ^ weighted_pic_mix64(
            epoch + UINT64_C(0x6a09e667f3bcc909)));
    const std::size_t lane =
        static_cast<std::size_t>(lane_hash % lanes);
    if (radius < mesh.edge.front()) {
        const double fraction = std::clamp(
            std::pow(radius / mesh.edge.front(), 3),
            0.0,
            std::nextafter(1.0, 0.0));
        const std::size_t radial_cell = std::min(
            inner - 1,
            static_cast<std::size_t>(fraction * inner));
        return radial_cell * lanes + lane;
    }
    return (inner + resolved_radial_cell(radius, mesh)) * lanes + lane;
}

// Each Monte Carlo lane owns 1/lanes of one physical shell volume. This
// volume factor compensates the probability that a pair occupies the same lane.
inline double collision_cell_volume(
    std::size_t cell,
    const WeightedPICRadialMesh& mesh,
    const WeightedPICSubhaloCollisionOptions& options) {
    const std::size_t inner =
        static_cast<std::size_t>(options.inner_collision_cells);
    const std::size_t lanes =
        static_cast<std::size_t>(options.collision_lanes);
    const std::size_t inner_lanes = inner * lanes;
    if (cell < inner_lanes) {
        return 4.0 * PI * std::pow(mesh.edge.front(), 3)
            / (3.0 * static_cast<double>(inner * lanes));
    }
    const std::size_t physical_cell = cell / lanes;
    return mesh.shell_volume[physical_cell - inner]
        / static_cast<double>(lanes);
}

inline double collision_cell_width(
    std::size_t cell,
    const WeightedPICRadialMesh& mesh,
    const WeightedPICSubhaloCollisionOptions& options) {
    const std::size_t inner =
        static_cast<std::size_t>(options.inner_collision_cells);
    const std::size_t lanes =
        static_cast<std::size_t>(options.collision_lanes);
    const std::size_t inner_lanes = inner * lanes;
    if (cell < inner_lanes) {
        const std::size_t radial_cell = cell / lanes;
        const double left = mesh.edge.front()
            * std::cbrt(static_cast<double>(radial_cell) / inner);
        const double right = mesh.edge.front()
            * std::cbrt(static_cast<double>(radial_cell + 1) / inner);
        return right - left;
    }
    const std::size_t resolved = cell / lanes - inner;
    return mesh.edge[resolved + 1] - mesh.edge[resolved];
}

inline double uniform(
    std::uint64_t seed,
    std::uint64_t epoch,
    std::uint64_t cell,
    std::uint64_t substep,
    std::uint64_t trial,
    std::uint64_t dimension) {
    const std::uint64_t counter =
        weighted_pic_mix64(epoch)
        ^ weighted_pic_mix64(cell + UINT64_C(0x100000001b3))
        ^ weighted_pic_mix64(substep + UINT64_C(0x9e3779b9))
        ^ weighted_pic_mix64(trial + UINT64_C(0x85ebca6b));
    return weighted_pic_counter_uniform(seed, counter, dimension);
}

inline Vec3 marker_velocity(
    const WeightedPICMarker& marker,
    double azimuth) {
    const double tangential_speed = std::sqrt(2.0 * marker.eta)
        / std::max(marker.r, 1.0e-300);
    return {
        marker.w,
        tangential_speed * std::cos(azimuth),
        tangential_speed * std::sin(azimuth),
    };
}

// Return a local Cartesian velocity to w and eta at its unchanged radius.
// At the exact origin only total speed remains meaningful in the spherical
// representation, and every finite angular momentum there is zero.
inline void install_velocity(
    WeightedPICMarker& marker,
    const Vec3& velocity) {
    if (marker.r == 0.0) {
        // At the origin a spherical distribution is isotropic.  Every finite
        // velocity has L=0; its speed is the outgoing radial velocity of the
        // subsequent orbit.  Preserve all three Cartesian kinetic components.
        marker.w = std::hypot(velocity[0], std::hypot(velocity[1], velocity[2]));
        marker.eta = 0.0;
        return;
    }
    marker.w = velocity[0];
    marker.eta = 0.5 * marker.r * marker.r
        * (velocity[1] * velocity[1] + velocity[2] * velocity[2]);
}

inline Vec3 isotropic_direction(double first, double second) {
    const double mu = 2.0 * first - 1.0;
    const double phi = 2.0 * PI * second;
    const double transverse = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    return {
        mu,
        transverse * std::cos(phi),
        transverse * std::sin(phi),
    };
}

// The direction of an origin marker is absent from the reduced (r,w,eta)
// representation.  Sample its isotropic Cartesian velocity for this pair;
// use that same vector in both the rate kernel and the accepted outcome.
inline Vec3 sample_origin_velocity(
    const WeightedPICMarker& marker,
    double azimuth,
    std::uint64_t seed,
    std::uint64_t epoch,
    std::uint64_t cell,
    std::uint64_t substep,
    std::uint64_t trial,
    std::uint64_t first_dimension) {
    if (marker.r > 0.0) {
        return marker_velocity(marker, azimuth);
    }
    return scale(isotropic_direction(
        uniform(seed, epoch, cell, substep, trial, first_dimension),
        uniform(seed, epoch, cell, substep, trial, first_dimension + 1)),
        std::abs(marker.w));
}

inline void accumulate(
    WeightedPICSubhaloCollisionDiagnostics& target,
    const WeightedPICSubhaloCollisionDiagnostics& source) {
    target.substeps += source.substeps;
    target.cell_visits += source.cell_visits;
    target.trial_pairs += source.trial_pairs;
    target.accepted_pairs += source.accepted_pairs;
    target.scattering_events += source.scattering_events;
    target.capture_events += source.capture_events;
    target.inner_cell_visits += source.inner_cell_visits;
    target.inner_scattering_events += source.inner_scattering_events;
    target.inner_capture_events += source.inner_capture_events;
    target.skipped_already_collided_trials +=
        source.skipped_already_collided_trials;
    target.spawned_markers += source.spawned_markers;
    target.removed_markers += source.removed_markers;
    target.nonlocal_cells += source.nonlocal_cells;
    target.represented_scattering_mass +=
        source.represented_scattering_mass;
    target.represented_capture_mass += source.represented_capture_mass;
    target.orbital_kinetic_energy_change +=
        source.orbital_kinetic_energy_change;
    target.internal_energy_change += source.internal_energy_change;
    target.maximum_pair_momentum_error = std::max(
        target.maximum_pair_momentum_error,
        source.maximum_pair_momentum_error);
    target.maximum_pair_total_energy_error = std::max(
        target.maximum_pair_total_energy_error,
        source.maximum_pair_total_energy_error);
    target.minimum_mean_free_path = std::min(
        target.minimum_mean_free_path, source.minimum_mean_free_path);
    target.minimum_collision_timescale = std::min(
        target.minimum_collision_timescale,
        source.minimum_collision_timescale);
    target.maximum_cell_width_over_mean_free_path = std::max(
        target.maximum_cell_width_over_mean_free_path,
        source.maximum_cell_width_over_mean_free_path);
    target.mean_free_path_pair_samples +=
        source.mean_free_path_pair_samples;
    target.minimum_orbital_velocity_dispersion = std::min(
        target.minimum_orbital_velocity_dispersion,
        source.minimum_orbital_velocity_dispersion);
    target.maximum_orbital_velocity_dispersion = std::max(
        target.maximum_orbital_velocity_dispersion,
        source.maximum_orbital_velocity_dispersion);
    target.maximum_weight_ratio = std::max(
        target.maximum_weight_ratio, source.maximum_weight_ratio);
    target.minimum_internal_velocity_dispersion = std::min(
        target.minimum_internal_velocity_dispersion,
        source.minimum_internal_velocity_dispersion);
    target.maximum_internal_velocity_dispersion = std::max(
        target.maximum_internal_velocity_dispersion,
        source.maximum_internal_velocity_dispersion);
    target.minimum_pair_velocity_scale = std::min(
        target.minimum_pair_velocity_scale,
        source.minimum_pair_velocity_scale);
    target.maximum_pair_velocity_scale = std::max(
        target.maximum_pair_velocity_scale,
        source.maximum_pair_velocity_scale);
}

}  // namespace weighted_pic_subhalo_collision_detail

// A successful call completes every cell through dt before compacting markers.
// On an exception, the caller must discard the partially advanced marker and
// sampler state.  The driver propagates the failure before committing event
// clocks or a checkpoint.  The workspace is scratch storage for the next call.
inline WeightedPICSubhaloCollisionDiagnostics
// This operator updates particles in place during its trial sequence. The
// caller commits event clocks only after success and discards the partially
// advanced trial on failure, preserving its diagnostic state separately.
apply_weighted_pic_subhalo_collision_interval(
    std::vector<WeightedPICMarker>& markers,
    const WeightedPICRadialMesh& mesh,
    double dt,
    const WeightedPICSubhaloCollisionOptions& options,
    WeightedPICCollisionSamplerState& sampler,
    WeightedPICSubhaloCollisionWorkspace& workspace,
    unsigned int requested_threads = 1) {
    using namespace weighted_pic_subhalo_collision_detail;
    validate_options(options);
    if (!(dt >= 0.0) || !std::isfinite(dt) || requested_threads == 0
        || mesh.shell_volume.empty()
        || mesh.edge.size() != mesh.shell_volume.size() + 1) {
        throw std::runtime_error(
            "Invalid weighted-PIC subhalo collision interval or mesh.");
    }
    WeightedPICSubhaloCollisionDiagnostics total;
    if (dt == 0.0 || options.strength == 0.0) {
        return total;
    }

    const double invariant_mass = weighted_pic_total_marker_mass(markers);
    long double initial_internal_energy = 0.0L;
    for (const WeightedPICMarker& marker : markers) {
        validate_weighted_pic_marker(marker, "Subhalo collision input");
        if (marker.state == WeightedPICMarkerState::active
            && marker.r == 0.0 && marker.eta != 0.0) {
            throw std::runtime_error(
                "A subhalo at the origin must have zero angular momentum.");
        }
        initial_internal_energy += marker.internal_energy;
        if (marker.state == WeightedPICMarkerState::active) {
            ++total.active_markers_before;
        }
    }
    const std::uint64_t epoch = sampler.next_counter++;
    const std::size_t inner_cell_count =
        static_cast<std::size_t>(options.inner_collision_cells)
        * static_cast<std::size_t>(options.collision_lanes);
    const std::size_t cell_count =
        collision_cell_count(mesh, options);
    const std::size_t physical_cell_count =
        static_cast<std::size_t>(options.inner_collision_cells)
        + mesh.shell_volume.size();
    const int cell_workers = std::max(
        1,
        std::min(
            static_cast<int>(requested_threads),
            static_cast<int>(cell_count)));
    // Radial cells are ordered from the dense cusp outward.  Contiguous
    // chunks would therefore put all expensive inner cells on the first
    // worker.  A cyclic assignment spreads adjacent Monte-Carlo lanes and
    // radial cells across workers without changing any per-cell operations
    // or random streams.
    auto parallel_for_collision_cells = [&](auto&& function) {
        parallel_for_chunks(
            0,
            cell_workers,
            cell_workers,
            [&](int begin, int end, int thread) {
                for (int worker = begin; worker < end; ++worker) {
                    for (std::size_t cell =
                             static_cast<std::size_t>(worker);
                         cell < cell_count;
                         cell += static_cast<std::size_t>(cell_workers)) {
                        function(cell, thread);
                    }
                }
            });
    };

    workspace.cell_members.resize(cell_count);
    for (auto& members : workspace.cell_members) {
        members.clear();
    }
    workspace.physical_cell_members.resize(physical_cell_count);
    for (auto& members : workspace.physical_cell_members) {
        members.clear();
    }
    workspace.cell_pair_order.resize(cell_count);
    for (auto& order : workspace.cell_pair_order) {
        order.clear();
    }
    workspace.cell_pair_cumulative_weight.resize(cell_count);
    for (auto& cumulative : workspace.cell_pair_cumulative_weight) {
        cumulative.clear();
    }
    workspace.cell_pair_prefix_root_mass.resize(cell_count);
    for (auto& prefix : workspace.cell_pair_prefix_root_mass) {
        prefix.clear();
    }
    workspace.cell_pair_prefix_mass.resize(cell_count);
    for (auto& prefix : workspace.cell_pair_prefix_mass) {
        prefix.clear();
    }
    workspace.cell_collided_generation.resize(cell_count);
    for (auto& generation : workspace.cell_collided_generation) {
        generation.clear();
    }
    workspace.cell_diagnostics.assign(
        cell_count, WeightedPICSubhaloCollisionDiagnostics{});
    workspace.physical_cell_diagnostics.assign(
        physical_cell_count, WeightedPICSubhaloCollisionDiagnostics{});
    workspace.remove_marker.assign(markers.size(), 0);
    for (std::size_t marker_index = 0; marker_index < markers.size();
         ++marker_index) {
        const WeightedPICMarker& marker = markers[marker_index];
        if (marker.state != WeightedPICMarkerState::active) {
            continue;
        }
        enforce_internal_state(marker, "Subhalo collision input");
        total.minimum_internal_velocity_dispersion = std::min(
            total.minimum_internal_velocity_dispersion,
            marker.internal_velocity_dispersion);
        total.maximum_internal_velocity_dispersion = std::max(
            total.maximum_internal_velocity_dispersion,
            marker.internal_velocity_dispersion);
        // Markers beyond the outer field edge have not yet been assigned a
        // physical local collision volume.  In contrast, the regular origin
        // core is covered by explicit equal-volume virtual collision shells.
        if (marker.r > mesh.edge.back()) {
            continue;
        }
        const std::size_t cell =
            collision_cell(marker.r, marker.id, epoch, mesh, options);
        workspace.cell_members[cell].push_back(marker_index);
        workspace.physical_cell_members[
            cell / static_cast<std::size_t>(options.collision_lanes)]
            .push_back(marker_index);
    }

    // Frank & Tokayer use lambda_mfp=1/[rho*(sigma/m)] after measuring rho
    // from each current radial shell.  Here the encounter area depends on
    // both marker masses and on relative speed, so the exact generalization is
    //
    //   lambda_mfp = <u> / [n_partner <u Sigma_ij(u)>],
    //
    // with n_partner=(N-1)/V for a finite cell.
    //
    // Evaluate both averages directly from the evolving marker distribution.
    // The physical radial cells are used here; Monte-Carlo lanes are only a
    // parallel sampling device and must not change a physical mean free path.
    // Measure locality from the evolving distribution over full physical cells,
    // combining all lanes. The same sampled <u> and <u*Sigma> define both mean free
    // path and physical encounter time with the finite partner density (N-1)/V.
    const int physical_workers = std::max(
        1,
        std::min(
            static_cast<int>(requested_threads),
            static_cast<int>(physical_cell_count)));
    parallel_for_chunks(
        0,
        static_cast<int>(physical_cell_count),
        physical_workers,
        [&](int begin, int end, int) {
            for (int physical = begin; physical < end; ++physical) {
                const std::size_t physical_cell =
                    static_cast<std::size_t>(physical);
                const auto& members =
                    workspace.physical_cell_members[physical_cell];
                auto& stats =
                    workspace.physical_cell_diagnostics[physical_cell];
                if (members.size() < options.minimum_cell_population) {
                    continue;
                }

                long double mass_sum = 0.0L;
                long double radial_momentum = 0.0L;
                long double speed_squared_moment = 0.0L;
                for (const std::size_t index : members) {
                    const WeightedPICMarker& marker = markers[index];
                    const long double mass = marker.mass;
                    const long double radius =
                        std::max(marker.r, 1.0e-300);
                    const long double tangential_speed_squared =
                        marker.eta == 0.0 ? 0.0L
                        : 2.0L * marker.eta / (radius * radius);
                    mass_sum += mass;
                    // Relative to any chosen Cartesian axis at the origin,
                    // <v_r>=0, <v_r^2>=w^2/3 and <v_t^2>=2w^2/3.
                    // w stores the full speed there, so the trace is w^2.
                    if (marker.r > 0.0) {
                        radial_momentum += mass * marker.w;
                    }
                    speed_squared_moment += mass
                        * (marker.w * marker.w
                           + tangential_speed_squared);
                }
                const long double mean_radial_velocity =
                    radial_momentum / mass_sum;
                const long double centred_three_dimensional_variance =
                    std::max(
                        0.0L,
                        speed_squared_moment / mass_sum
                            - mean_radial_velocity
                                * mean_radial_velocity);
                const double orbital_dispersion = std::sqrt(
                    static_cast<double>(
                        centred_three_dimensional_variance / 3.0L));
                stats.minimum_orbital_velocity_dispersion =
                    orbital_dispersion;
                stats.maximum_orbital_velocity_dispersion =
                    orbital_dispersion;

                const long double pair_count =
                    0.5L
                    * static_cast<long double>(members.size())
                    * static_cast<long double>(members.size() - 1);
                const std::uint64_t requested_samples =
                    physical_cell
                            < static_cast<std::size_t>(
                                options.inner_collision_cells)
                        ? UINT64_C(4096) : UINT64_C(256);
                const std::uint64_t samples =
                    static_cast<std::uint64_t>(std::min(
                        pair_count,
                        static_cast<long double>(requested_samples)));
                if (samples == 0) {
                    continue;
                }

                long double relative_speed_sum = 0.0L;
                long double collision_kernel_sum = 0.0L;
                auto sample_pair = [&](std::size_t first_local,
                                       std::size_t second_local,
                                       std::uint64_t sample) {
                    const WeightedPICMarker& first =
                        markers[members[first_local]];
                    const WeightedPICMarker& second =
                        markers[members[second_local]];
                    const double relative_azimuth =
                        2.0 * PI * uniform(
                            sampler.seed,
                            epoch,
                            physical_cell,
                            UINT64_C(0x6d66705f6466),
                            sample,
                            31);
                    const double first_tangential_speed =
                        std::sqrt(2.0 * first.eta)
                        / std::max(first.r, 1.0e-300);
                    const double second_tangential_speed =
                        std::sqrt(2.0 * second.eta)
                        / std::max(second.r, 1.0e-300);
                    const double radial_difference =
                        first.w - second.w;
                    double relative_speed = std::sqrt(std::max(
                        0.0,
                        radial_difference * radial_difference
                            + first_tangential_speed
                                * first_tangential_speed
                            + second_tangential_speed
                                * second_tangential_speed
                            - 2.0 * first_tangential_speed
                                * second_tangential_speed
                                * std::cos(relative_azimuth)));
                    if (first.r == 0.0 || second.r == 0.0) {
                        const Vec3 first_velocity = sample_origin_velocity(
                            first, 0.0, sampler.seed, epoch, physical_cell,
                            UINT64_C(0x6d66705f6466), sample, 34);
                        const Vec3 second_velocity = sample_origin_velocity(
                            second, relative_azimuth, sampler.seed, epoch,
                            physical_cell, UINT64_C(0x6d66705f6466), sample, 36);
                        relative_speed = norm(subtract(first_velocity, second_velocity));
                    }
                    const double pair_sigma =
                        pair_velocity_scale(first, second);
                    const double scattering =
                        weighted_pic_subhalo_scattering_cross_section(
                            relative_speed,
                            first.mass,
                            second.mass,
                            pair_sigma,
                            options);
                    const double capture =
                        weighted_pic_subhalo_capture_cross_section(
                            relative_speed,
                            first.mass,
                            second.mass,
                            pair_sigma,
                            options);
                    relative_speed_sum += relative_speed;
                    collision_kernel_sum += relative_speed
                        * (scattering + capture);
                };

                if (pair_count
                    <= static_cast<long double>(requested_samples)) {
                    std::uint64_t sample = 0;
                    for (std::size_t first = 0;
                         first < members.size();
                         ++first) {
                        for (std::size_t second = first + 1;
                             second < members.size();
                             ++second) {
                            sample_pair(first, second, sample++);
                        }
                    }
                } else {
                    for (std::uint64_t sample = 0;
                         sample < samples;
                         ++sample) {
                        const std::size_t population = members.size();
                        const std::size_t first = std::min(
                            population - 1,
                            static_cast<std::size_t>(
                                uniform(
                                    sampler.seed,
                                    epoch,
                                    physical_cell,
                                    UINT64_C(0x6d66705f6466),
                                    sample,
                                    32)
                                * static_cast<double>(population)));
                        std::size_t second = std::min(
                            population - 2,
                            static_cast<std::size_t>(
                                uniform(
                                    sampler.seed,
                                    epoch,
                                    physical_cell,
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

                stats.mean_free_path_pair_samples = samples;
                const std::size_t first_lane =
                    physical_cell
                    * static_cast<std::size_t>(options.collision_lanes);
                const double physical_volume =
                    collision_cell_volume(first_lane, mesh, options)
                    * static_cast<double>(options.collision_lanes);
                // One object has only N-1 possible partners.  Using the
                // finite-cell partner density makes the measured MFP and
                // collision clock exactly consistent with the C(N,2) trial
                // law; N/V is recovered in the continuum limit.
                const double partner_number_density =
                    static_cast<double>(members.size() - 1)
                    / physical_volume;
                stats.minimum_mean_free_path =
                    weighted_pic_subhalo_mean_free_path_from_pair_samples(
                        partner_number_density,
                        relative_speed_sum,
                        collision_kernel_sum);
                // The same sampled velocity-dependent kernel that defines
                // lambda_mfp also gives the local physical waiting time:
                //
                //   nu_coll = n_partner <u Sigma_tot(u)>,
                //   t_coll  = 1/nu_coll = lambda_mfp/<u>.
                //
                // Retaining the direct rate expression avoids dividing two
                // separately rounded Monte-Carlo means.
                const long double collision_frequency =
                    static_cast<long double>(partner_number_density)
                    * collision_kernel_sum
                    / static_cast<long double>(samples);
                if (collision_frequency > 0.0L
                    && std::isfinite(
                        static_cast<double>(collision_frequency))) {
                    stats.minimum_collision_timescale =
                        static_cast<double>(1.0L / collision_frequency);
                }
                if (std::isfinite(stats.minimum_mean_free_path)) {
                    stats.maximum_cell_width_over_mean_free_path =
                        collision_cell_width(first_lane, mesh, options)
                        / stats.minimum_mean_free_path;
                    if (stats.maximum_cell_width_over_mean_free_path > 1.0) {
                        stats.nonlocal_cells = 1;
                    }
                }
            }
        });
    for (const auto& stats : workspace.physical_cell_diagnostics) {
        accumulate(total, stats);
    }

    std::vector<double> cell_maximum_weight(cell_count, 0.0);
    std::vector<double> cell_minimum_weight(
        cell_count, std::numeric_limits<double>::infinity());
    std::vector<double> cell_minimum_sigma(
        cell_count, std::numeric_limits<double>::infinity());
    std::vector<double> cell_maximum_sigma(cell_count, 0.0);
    std::vector<double> cell_majorant(cell_count, 0.0);
    std::vector<double> cell_frequency(cell_count, 0.0);
    std::vector<unsigned char> cell_dirty(cell_count, 1);
    std::vector<double> cell_elapsed(cell_count, 0.0);
    std::vector<double> cell_subdt(cell_count, 0.0);
    std::vector<std::uint64_t> cell_substeps(cell_count, 0);

    // The proposal and accepted-pair updates access only one cell's marker
    // indices and scratch storage.  Their arithmetic and random counters are
    // shared by the global-clock and independent-cell schedules below.
    // After capture, remove absorbed partners from this cell and rebuild its
    // proposal distribution. Elastic orbital scattering leaves masses and internal
    // dispersions fixed, so its pair-area majorants remain reusable.
    auto prepare_cell = [&](std::size_t cell) {
        if (cell_dirty[cell] == 0) {
            return;
        }
        auto& members = workspace.cell_members[cell];
        members.erase(
            std::remove_if(
                members.begin(),
                members.end(),
                [&](std::size_t index) {
                    return workspace.remove_marker[index] != 0;
                }),
            members.end());
        cell_maximum_weight[cell] = 0.0;
        cell_minimum_weight[cell] =
            std::numeric_limits<double>::infinity();
        cell_minimum_sigma[cell] =
            std::numeric_limits<double>::infinity();
        cell_maximum_sigma[cell] = 0.0;
        cell_majorant[cell] = 0.0;
        cell_frequency[cell] = 0.0;
        cell_dirty[cell] = 0;
        for (const std::size_t index : members) {
            const WeightedPICMarker& marker = markers[index];
            cell_maximum_weight[cell] = std::max(
                cell_maximum_weight[cell], marker.mass);
            cell_minimum_weight[cell] = std::min(
                cell_minimum_weight[cell], marker.mass);
            cell_minimum_sigma[cell] = std::min(
                cell_minimum_sigma[cell],
                marker.internal_velocity_dispersion);
            cell_maximum_sigma[cell] = std::max(
                cell_maximum_sigma[cell],
                marker.internal_velocity_dispersion);
        }
        auto& pair_order = workspace.cell_pair_order[cell];
        auto& pair_cumulative =
            workspace.cell_pair_cumulative_weight[cell];
        auto& prefix_root =
            workspace.cell_pair_prefix_root_mass[cell];
        auto& prefix_mass =
            workspace.cell_pair_prefix_mass[cell];
        pair_order = members;
        pair_cumulative.clear();
        prefix_root.clear();
        prefix_mass.clear();
        if (members.size() < options.minimum_cell_population
            || !(cell_maximum_sigma[cell] > 0.0)) {
            return;
        }

        // Sort only the proposal index, never the marker array.  For
        // i<j, sigma_j is then max(sigma_i,sigma_j), and the safe pair
        // weight is
        //   B_ij = C sigma_j (sqrt(m_i)+sqrt(m_j))^2.
        // Its sum over i<j is available from prefix sums of sqrt(m)
        // and m.  This constructs the exact proposal distribution in
        // O(N log N), rather than materializing O(N^2) pairs.
        // The key (internal dispersion, unique marker id) is a total
        // ordering, so stability cannot affect the proposal order.
        // std::sort avoids the extra merge buffer and is therefore
        // cheaper in the multi-million-marker cusp cells.
        std::sort(
            pair_order.begin(),
            pair_order.end(),
            [&](std::size_t first, std::size_t second) {
                const WeightedPICMarker& a = markers[first];
                const WeightedPICMarker& b = markers[second];
                if (a.internal_velocity_dispersion
                    != b.internal_velocity_dispersion) {
                    return a.internal_velocity_dispersion
                        < b.internal_velocity_dispersion;
                }
                return a.id < b.id;
            });
        // Sorting by internal dispersion identifies the larger dispersion in each
        // pair. Prefix sums of mass and sqrt(mass) then sum all pair bounds without
        // storing the quadratic number of individual pairs.
        const std::size_t population = pair_order.size();
        prefix_root.assign(population + 1, 0.0L);
        prefix_mass.assign(population + 1, 0.0L);
        pair_cumulative.assign(population, 0.0L);
        long double pair_weight_sum = 0.0L;
        for (std::size_t high = 0; high < population; ++high) {
            const WeightedPICMarker& marker =
                markers[pair_order[high]];
            const long double root_mass = std::sqrt(marker.mass);
            const long double lower_area_sum =
                prefix_mass[high]
                + 2.0L * root_mass * prefix_root[high]
                + static_cast<long double>(high)
                    * root_mass * root_mass;
            pair_weight_sum +=
                marker.internal_velocity_dispersion * lower_area_sum;
            pair_cumulative[high] = pair_weight_sum;
            prefix_root[high + 1] =
                prefix_root[high] + root_mass;
            prefix_mass[high + 1] =
                prefix_mass[high] + marker.mass;
        }

        const double pair_kernel_scale =
            0.25
            * weighted_pic_subhalo_u_sigma_majorant(1.0, options)
            * (1.0
               + 128.0
                   * std::numeric_limits<double>::epsilon());
        cell_majorant[cell] = static_cast<double>(
            pair_kernel_scale * pair_weight_sum);

        // Bound the largest trial frequency of any one marker.  This
        // is the physical quantity controlled by collision_cfl when
        // one accepted event per marker per substep is enforced.
        long double suffix_sigma = 0.0L;
        long double suffix_sigma_root = 0.0L;
        long double suffix_sigma_mass = 0.0L;
        long double maximum_marker_bound = 0.0L;
        for (std::size_t reverse = population;
             reverse-- > 0;) {
            const WeightedPICMarker& marker =
                markers[pair_order[reverse]];
            const long double root_mass = std::sqrt(marker.mass);
            const long double lower =
                marker.internal_velocity_dispersion
                * (prefix_mass[reverse]
                   + 2.0L * root_mass * prefix_root[reverse]
                   + static_cast<long double>(reverse)
                       * root_mass * root_mass);
            const long double upper =
                root_mass * root_mass * suffix_sigma
                + 2.0L * root_mass * suffix_sigma_root
                + suffix_sigma_mass;
            maximum_marker_bound = std::max(
                maximum_marker_bound, lower + upper);
            suffix_sigma += marker.internal_velocity_dispersion;
            suffix_sigma_root +=
                marker.internal_velocity_dispersion * root_mass;
            suffix_sigma_mass +=
                marker.internal_velocity_dispersion * marker.mass;
        }
        const double volume =
            collision_cell_volume(cell, mesh, options);
        cell_frequency[cell] = static_cast<double>(
            pair_kernel_scale * maximum_marker_bound / volume);
    };
    auto advance_cell = [&](std::size_t cell, double applied_subdt,
                            std::uint64_t local_substep) {
        const std::vector<std::size_t>& members =
            workspace.cell_members[cell];
        if (members.size() < options.minimum_cell_population) {
            return;
        }
        const std::vector<std::size_t>& pair_order =
            workspace.cell_pair_order[cell];
        const std::vector<long double>& pair_cumulative =
            workspace.cell_pair_cumulative_weight[cell];
        const std::vector<long double>& prefix_root =
            workspace.cell_pair_prefix_root_mass[cell];
        const std::vector<long double>& prefix_mass =
            workspace.cell_pair_prefix_mass[cell];
        auto& collided_generation =
            workspace.cell_collided_generation[cell];
        if (collided_generation.size() != pair_order.size()) {
            collided_generation.resize(pair_order.size(), 0);
        }
        const std::uint32_t current_generation =
            static_cast<std::uint32_t>(local_substep + 1);
        WeightedPICSubhaloCollisionDiagnostics& stats =
            workspace.cell_diagnostics[cell];
        // Elastic scattering changes only orbital velocity.  The
        // pair proposal depends on marker mass and internal
        // dispersion, so only a capture can invalidate it for the
        // next NTC substep.
        bool proposal_changed = false;
        ++stats.cell_visits;
        if (cell < inner_cell_count) {
            ++stats.inner_cell_visits;
        }
        const double maximum_weight =
            cell_maximum_weight[cell];
        const double minimum_weight =
            cell_minimum_weight[cell];
        stats.minimum_internal_velocity_dispersion = std::min(
            stats.minimum_internal_velocity_dispersion,
            cell_minimum_sigma[cell]);
        stats.maximum_internal_velocity_dispersion = std::max(
            stats.maximum_internal_velocity_dispersion,
            cell_maximum_sigma[cell]);
        if (!(maximum_weight > 0.0)
            || !(minimum_weight > 0.0)) {
            return;
        }
        stats.maximum_weight_ratio = std::max(
            stats.maximum_weight_ratio,
            maximum_weight / minimum_weight);
        const double volume =
            collision_cell_volume(cell, mesh, options);

        // The total pair-majorant sum times dt/volume gives the expected proposal
        // count. Stochastic rounding preserves that expected count for a finite substep.
        const long double expected_trials =
            static_cast<long double>(cell_majorant[cell])
            * applied_subdt / volume;
        if (expected_trials
            > static_cast<long double>(
                std::numeric_limits<std::uint64_t>::max() - 1)) {
            throw std::runtime_error(
                "Subhalo collision trial count overflows uint64.");
        }
        const double rounding = uniform(
            sampler.seed,
            epoch,
            cell,
            local_substep,
            UINT64_C(0xffffffffffffffff),
            0);
        const std::uint64_t trials =
            static_cast<std::uint64_t>(
                std::floor(expected_trials + rounding));
        for (std::uint64_t trial = 0; trial < trials; ++trial) {
            ++stats.trial_pairs;
            std::pair<std::size_t, std::size_t> positions;
            const std::size_t population = pair_order.size();
            const long double group_draw =
                uniform(
                    sampler.seed, epoch, cell,
                    local_substep, trial, 1)
                * pair_cumulative.back();
            auto group = std::upper_bound(
                pair_cumulative.begin() + 1,
                pair_cumulative.end(),
                group_draw);
            positions.second = std::min(
                population - 1,
                static_cast<std::size_t>(
                    group - pair_cumulative.begin()));
            const WeightedPICMarker& high_marker =
                markers[pair_order[positions.second]];
            const long double high_root =
                std::sqrt(high_marker.mass);
            const long double mass_component =
                prefix_mass[positions.second];
            const long double root_component =
                2.0L * high_root
                * prefix_root[positions.second];
            const long double uniform_component =
                static_cast<long double>(positions.second)
                * high_root * high_root;
            long double conditional_draw =
                uniform(
                    sampler.seed, epoch, cell,
                    local_substep, trial, 2)
                * (mass_component
                   + root_component
                   + uniform_component);
            if (conditional_draw < mass_component) {
                const auto selected = std::upper_bound(
                    prefix_mass.begin() + 1,
                    prefix_mass.begin() + positions.second + 1,
                    conditional_draw);
                positions.first = static_cast<std::size_t>(
                    selected - prefix_mass.begin() - 1);
            } else if (
                (conditional_draw -= mass_component)
                < root_component) {
                const long double root_draw =
                    conditional_draw / (2.0L * high_root);
                const auto selected = std::upper_bound(
                    prefix_root.begin() + 1,
                    prefix_root.begin() + positions.second + 1,
                    root_draw);
                positions.first = static_cast<std::size_t>(
                    selected - prefix_root.begin() - 1);
            } else {
                conditional_draw -= root_component;
                positions.first = std::min(
                    positions.second - 1,
                    static_cast<std::size_t>(
                        conditional_draw
                        / (high_root * high_root)));
            }
            const std::size_t first_local =
                positions.first;
            const std::size_t second_local =
                positions.second;
            if (collided_generation[first_local]
                    == current_generation
                || collided_generation[second_local]
                    == current_generation) {
                ++stats.skipped_already_collided_trials;
                continue;
            }
            const std::size_t first_index =
                pair_order[first_local];
            const std::size_t second_index =
                pair_order[second_local];
            if (workspace.remove_marker[first_index] != 0
                || workspace.remove_marker[second_index] != 0) {
                continue;
            }
            WeightedPICMarker& first = markers[first_index];
            WeightedPICMarker& second = markers[second_index];
            const double relative_azimuth =
                2.0 * PI * uniform(
                    sampler.seed, epoch, cell,
                    local_substep, trial, 3);
            const double first_tangential_speed =
                std::sqrt(2.0 * first.eta)
                / std::max(first.r, 1.0e-300);
            const double second_tangential_speed =
                std::sqrt(2.0 * second.eta)
                / std::max(second.r, 1.0e-300);
            const double radial_difference =
                first.w - second.w;
            const double relative_speed_squared = std::max(
                0.0,
                radial_difference * radial_difference
                    + first_tangential_speed
                        * first_tangential_speed
                    + second_tangential_speed
                        * second_tangential_speed
                    - 2.0 * first_tangential_speed
                        * second_tangential_speed
                        * std::cos(relative_azimuth));
            double relative_speed = std::sqrt(relative_speed_squared);
            const bool origin_pair = first.r == 0.0 || second.r == 0.0;
            Vec3 origin_first_velocity{}, origin_second_velocity{};
            if (origin_pair) {
                const double azimuth = 2.0 * PI * uniform(
                    sampler.seed, epoch, cell, local_substep, trial, 4);
                origin_first_velocity = sample_origin_velocity(
                    first, azimuth, sampler.seed, epoch, cell,
                    local_substep, trial, 10);
                origin_second_velocity = sample_origin_velocity(
                    second, azimuth + relative_azimuth, sampler.seed, epoch,
                    cell, local_substep, trial, 12);
                relative_speed = norm(subtract(
                    origin_first_velocity, origin_second_velocity));
            }
            const double pair_sigma =
                pair_velocity_scale(first, second);
            stats.minimum_pair_velocity_scale = std::min(
                stats.minimum_pair_velocity_scale, pair_sigma);
            stats.maximum_pair_velocity_scale = std::max(
                stats.maximum_pair_velocity_scale, pair_sigma);
            const double scattering =
                weighted_pic_subhalo_scattering_cross_section(
                    relative_speed,
                    first.mass,
                    second.mass,
                    pair_sigma,
                    options);
            const double capture =
                weighted_pic_subhalo_capture_cross_section(
                    relative_speed,
                    first.mass,
                    second.mass,
                    pair_sigma,
                    options);
            const double kernel =
                relative_speed * (scattering + capture);
            const double pair_majorant =
                weighted_pic_subhalo_pair_u_sigma_majorant(
                    first.mass,
                    second.mass,
                    first.internal_velocity_dispersion,
                    second.internal_velocity_dispersion,
                    options);
            // Importance selection used this particular pair bound. Dividing its
            // physical u*(Sigma_scatter+Sigma_capture) by that same bound preserves the
            // unequal-object rate and exposes any invalid majorant.
            const double kernel_acceptance =
                kernel / pair_majorant;
            if (kernel_acceptance
                > 1.0
                    + 1024.0
                        * std::numeric_limits<double>::epsilon()) {
                std::ostringstream message;
                message << std::setprecision(17)
                        << "Marker-dependent collision majorant "
                           "was violated: acceptance="
                        << kernel_acceptance << ", u="
                        << relative_speed
                        << ", sigma_pair=" << pair_sigma
                        << ", sigma_pair_bound="
                        << std::max(
                               first.internal_velocity_dispersion,
                               second.internal_velocity_dispersion)
                        << ", pair_majorant="
                        << pair_majorant << '.';
                throw std::runtime_error(message.str());
            }
            if (uniform(
                    sampler.seed, epoch, cell,
                    local_substep,
                    trial, 5)
                >= std::min(1.0, kernel_acceptance)) {
                continue;
            }
            collided_generation[first_local] = current_generation;
            collided_generation[second_local] = current_generation;
            ++stats.accepted_pairs;
            const double first_azimuth = 2.0 * PI * uniform(
                sampler.seed, epoch, cell,
                local_substep, trial, 4);
            const double second_azimuth =
                first_azimuth + relative_azimuth;
            const Vec3 first_velocity = origin_pair ? origin_first_velocity
                : marker_velocity(first, first_azimuth);
            const Vec3 second_velocity = origin_pair ? origin_second_velocity
                : marker_velocity(second, second_azimuth);
            const double first_mass = first.mass;
            const double second_mass = second.mass;
            const double pair_mass = first_mass + second_mass;
            const double reduced_mass =
                first_mass * second_mass / pair_mass;
            stats.maximum_weight_ratio = std::max(
                stats.maximum_weight_ratio,
                std::max(first_mass, second_mass)
                    / std::min(first_mass, second_mass));
            const Vec3 centre = scale(
                add(
                    scale(first_velocity, first_mass),
                    scale(second_velocity, second_mass)),
                1.0 / pair_mass);
            const double internal_before =
                first.internal_energy + second.internal_energy;
            const double kinetic_before =
                0.5 * first_mass
                    * dot(first_velocity, first_velocity)
                + 0.5 * second_mass
                    * dot(second_velocity, second_velocity);
            const Vec3 momentum_before = add(
                scale(first_velocity, first_mass),
                scale(second_velocity, second_mass));

            const double channel_draw = uniform(
                sampler.seed, epoch, cell,
                local_substep, trial, 6);
            // After accepting the joint encounter, draw one exclusive channel in
            // proportion to its physical cross section. Both channels share the same
            // normalization and incoming velocity realization.
            const bool is_capture = capture > 0.0
                && channel_draw < capture / (scattering + capture);
            if (is_capture) {
                ++stats.capture_events;
                if (cell < inner_cell_count) {
                    ++stats.inner_capture_events;
                }
                stats.represented_capture_mass += pair_mass;
                // For capture, the centre-of-mass velocity preserves total momentum. The
                // relative kinetic energy mu*u^2/2 becomes internal heat, and the mass-weighted
                // remnant radius stays inside the original physical collision cell.
                const double lost_orbital_energy =
                    0.5 * reduced_mass * relative_speed
                    * relative_speed;
                const double merged_internal =
                    internal_before + lost_orbital_energy;
                const double merged_radius =
                    (first_mass * first.r
                     + second_mass * second.r)
                    / pair_mass;
                first.mass = pair_mass;
                first.r = merged_radius;
                first.internal_energy = merged_internal;
                update_internal_velocity_dispersion(first);
                install_velocity(first, centre);
                workspace.remove_marker[second_index] = 1;
                proposal_changed = true;
                ++stats.removed_markers;
                enforce_internal_state(
                    first, "Subhalo capture remnant");
                stats.maximum_internal_velocity_dispersion =
                    std::max(
                        stats.maximum_internal_velocity_dispersion,
                        first.internal_velocity_dispersion);
                const double kinetic_after =
                    0.5 * pair_mass * dot(centre, centre);
                const Vec3 momentum_after =
                    scale(centre, pair_mass);
                stats.orbital_kinetic_energy_change +=
                    kinetic_after - kinetic_before;
                stats.internal_energy_change +=
                    merged_internal - internal_before;
                stats.maximum_pair_momentum_error = std::max(
                    stats.maximum_pair_momentum_error,
                    norm(subtract(
                        momentum_after, momentum_before)));
                stats.maximum_pair_total_energy_error = std::max(
                    stats.maximum_pair_total_energy_error,
                    std::abs(
                        kinetic_after + merged_internal
                        - kinetic_before - internal_before));
            } else {
                ++stats.scattering_events;
                if (cell < inner_cell_count) {
                    ++stats.inner_scattering_events;
                }
                stats.represented_scattering_mass += pair_mass;
                const Vec3 direction = isotropic_direction(
                    uniform(
                        sampler.seed, epoch, cell,
                        local_substep,
                        trial, 7),
                    uniform(
                        sampler.seed, epoch, cell,
                        local_substep,
                        trial, 8));
                const Vec3 relative_after =
                    scale(direction, relative_speed);
                const Vec3 first_after = add(
                    centre,
                    scale(relative_after,
                          second_mass / pair_mass));
                const Vec3 second_after = subtract(
                    centre,
                    scale(relative_after,
                          first_mass / pair_mass));
                install_velocity(first, first_after);
                install_velocity(second, second_after);
                enforce_internal_state(
                    first, "First scattered subhalo");
                enforce_internal_state(
                    second, "Second scattered subhalo");
                const double kinetic_after =
                    0.5 * first_mass
                        * dot(first_after, first_after)
                    + 0.5 * second_mass
                        * dot(second_after, second_after);
                const Vec3 momentum_after = add(
                    scale(first_after, first_mass),
                    scale(second_after, second_mass));
                stats.orbital_kinetic_energy_change +=
                    kinetic_after - kinetic_before;
                stats.maximum_pair_momentum_error = std::max(
                    stats.maximum_pair_momentum_error,
                    norm(subtract(
                        momentum_after, momentum_before)));
                stats.maximum_pair_total_energy_error = std::max(
                    stats.maximum_pair_total_energy_error,
                    std::abs(kinetic_after - kinetic_before));
            }
        }
        if (proposal_changed) {
            cell_dirty[cell] = 1;
        }
    };

    // The maximum single-object proposal frequency bounds a substep when each
    // object receives at most one accepted encounter. Adaptive cells consume the
    // whole interval independently; the reference schedule shares the fastest clock.
    std::uint64_t substep = 0;
    if (options.adaptive_cell_timestep) {
        // Once membership is assigned, each cell advances through the complete
        // interval independently.  Orbital drift is paused; a capture's
        // mass-weighted remnant radius stays inside the assigned radial cell.
        // The seed, epoch, cell, local substep and trial identify each random
        // draw.  No floating-point sum is reordered within a cell, and
        // diagnostics retain their final reduction in increasing cell order.
        std::vector<std::exception_ptr> cell_errors(cell_count);
        parallel_for_collision_cells([&](std::size_t cell, int) {
            try {
                while (cell_elapsed[cell] < dt) {
                    prepare_cell(cell);
                    if (cell_substeps[cell]
                        >= static_cast<std::uint64_t>(options.maximum_substeps)) {
                        std::ostringstream message;
                        message << "Adaptive subhalo collision cell " << cell
                                << " exceeded maximum_substeps="
                                << options.maximum_substeps << "; elapsed="
                                << cell_elapsed[cell] << " of interval=" << dt
                                << ", marker_trial_frequency="
                                << cell_frequency[cell] << '.';
                        throw std::runtime_error(message.str());
                    }
                    const double local_remaining = dt - cell_elapsed[cell];
                    cell_subdt[cell] = cell_frequency[cell] > 0.0
                        ? std::min(local_remaining,
                                   options.collision_cfl / cell_frequency[cell])
                        : local_remaining;
                    if (!(cell_subdt[cell] > 0.0)
                        || !std::isfinite(cell_subdt[cell])) {
                        throw std::runtime_error(
                            "Adaptive subhalo cell timestep is not positive and finite.");
                    }
                    advance_cell(cell, cell_subdt[cell], cell_substeps[cell]);
                    cell_elapsed[cell] = cell_subdt[cell] >= local_remaining
                        ? dt : cell_elapsed[cell] + cell_subdt[cell];
                    ++cell_substeps[cell];
                }
            } catch (...) {
                // Finish the remaining cells up to the same configured cap.
                // In a maximum-substep failure this reproduces the original
                // lockstep failure state and first failing cell exactly.
                cell_errors[cell] = std::current_exception();
            }
        });
        for (const auto& error : cell_errors) {
            if (error != nullptr) {
                std::rethrow_exception(error);
            }
        }
        substep = *std::max_element(cell_substeps.begin(), cell_substeps.end());
    } else {
        double elapsed = 0.0;
        double previous_maximum_frequency = 0.0;
        double previous_substep = 0.0;
        while (elapsed < dt) {
            if (substep >= static_cast<std::uint64_t>(options.maximum_substeps)) {
                std::ostringstream message;
                message << "Subhalo collisions exceeded maximum_substeps="
                        << options.maximum_substeps
                        << "; elapsed=" << elapsed << " of interval=" << dt
                        << ", previous_maximum_marker_trial_frequency="
                        << previous_maximum_frequency
                        << ", previous_substep=" << previous_substep
                        << ". Reduce the collision interval only if this "
                           "pair-specific physical-rate bound is verified.";
                throw std::runtime_error(message.str());
            }
            parallel_for_collision_cells([&](std::size_t cell, int) {
                prepare_cell(cell);
            });
            const double maximum_frequency = *std::max_element(
                cell_frequency.begin(), cell_frequency.end());
            const double remaining = dt - elapsed;
            const double subdt = maximum_frequency > 0.0
                ? std::min(remaining, options.collision_cfl / maximum_frequency)
                : remaining;
            if (!(subdt > 0.0) || !std::isfinite(subdt)) {
                throw std::runtime_error(
                    "Subhalo collision substep is not positive and finite.");
            }
            previous_substep = subdt;
            previous_maximum_frequency = maximum_frequency;
            parallel_for_collision_cells([&](std::size_t cell, int) {
                advance_cell(cell, subdt, substep);
            });
            elapsed = subdt >= remaining ? dt : elapsed + subdt;
            ++substep;
        }
    }

    // Each cell has one worker at a time, so diagnostics can accumulate in
    // place throughout the operator call.  Reducing them once here replaces
    // a full cell-array zero and reduction on every NTC substep without
    // changing any collision draw or marker update.
    for (const auto& stats : workspace.cell_diagnostics) {
        accumulate(total, stats);
    }
    total.substeps = substep;

    // Compact absorbed partners only after every cell completes. Stable retained
    // array order and persistent IDs preserve the surviving physical population.
    if (total.removed_markers != 0) {
        std::vector<WeightedPICMarker> compacted;
        compacted.reserve(markers.size() - total.removed_markers);
        for (std::size_t index = 0; index < markers.size(); ++index) {
            if (workspace.remove_marker[index] == 0) {
                compacted.push_back(markers[index]);
            }
        }
        markers.swap(compacted);
    }
    long double final_internal_energy = 0.0L;
    for (const WeightedPICMarker& marker : markers) {
        validate_weighted_pic_marker(marker, "Subhalo collision output");
        final_internal_energy += marker.internal_energy;
        if (marker.state == WeightedPICMarkerState::active) {
            enforce_internal_state(marker, "Subhalo collision output");
            ++total.active_markers_after;
            total.minimum_internal_velocity_dispersion = std::min(
                total.minimum_internal_velocity_dispersion,
                marker.internal_velocity_dispersion);
            total.maximum_internal_velocity_dispersion = std::max(
                total.maximum_internal_velocity_dispersion,
                marker.internal_velocity_dispersion);
        }
    }
    // Close mass and internal-energy bookkeeping independently from the
    // individual pair residuals before returning the completed collision interval.
    const double final_mass = weighted_pic_total_marker_mass(markers);
    if (std::abs(final_mass - invariant_mass)
        > 256.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, invariant_mass)) {
        throw std::runtime_error(
            "Subhalo collisions failed to conserve represented mass.");
    }
    const double measured_internal_change = static_cast<double>(
        final_internal_energy - initial_internal_energy);
    if (std::abs(measured_internal_change - total.internal_energy_change)
        > 1024.0 * std::numeric_limits<double>::epsilon()
            * std::max({
                1.0,
                std::abs(measured_internal_change),
                std::abs(total.internal_energy_change)})) {
        throw std::runtime_error(
            "Subhalo collision internal-energy bookkeeping drifted.");
    }
    return total;
}

}  // namespace ev
