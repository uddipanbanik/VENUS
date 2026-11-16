#pragma once

#include "ev/central_drift.hpp"

// Weighted particle-mesh representation for static or expanding spherical
// Vlasov-Poisson.  Collisionless transport lives here; the optional PIC
// collision extension is kept in weighted_vlasov_pic_collisions.hpp.
//
// Representation and evolution order:
//   1. Marker/state/restart structures define the complete persistent state.
//   2. Initializers integrate source-cell mass and sample quiet in-cell
//      coordinates without changing that represented mass.
//   3. CIC/TSC deposition in q=r^3/3 forms shell moments and effective marker
//      counts, with deterministic thread-ordered reductions.
//   4. Static and expanding kick-drift-kick routines advance exact free
//      central-force drifts between endpoint gravitational kicks.
//   5. Boundary events update active, delayed, escaped, and hidden states.
//   6. Restart I/O checkpoints marker IDs, internal energy, random streams,
//      event ages, time, and step through an atomic versioned format.
//
// Marker mass is the physical mass represented by one numerical marker and
// therefore also its quadrature weight in the field solve.  The present
// coarse-object collision closure treats that complete represented mass as
// one collision partner; it is not the microscopic constituent-particle mass.
// Keep these two meanings (represented coarse-object mass versus microscopic
// particle mass) explicit whenever marker masses are unequal.

#include "ev/delayed_inner_orbit_carrier.hpp"
#include "ev/parallel.hpp"
#include "ev/vlasov_poisson.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <type_traits>

namespace ev {

// Persistent spherical Vlasov--Poisson particles carry positive represented
// mass, physical radius, radial peculiar velocity and eta=L^2/2. Source-cell
// quadrature creates the particles; radial clouds source the field; orbits evolve
// them continuously. Restarts retain identities, random counters and event ages.
// Collision modules separately define how a represented mass enters their laws.
enum class WeightedPICMarkerState : std::uint8_t {
    active = 0,
    delayed_inner = 1,
    escaped = 2,
};

/// One weighted phase-space marker.  Mass is invariant under collisionless
/// transport; eta may change in a collision.  internal_energy is the absolute
/// (rather than specific) unresolved energy carried by this marker.
/// internal_velocity_dispersion is the one-dimensional Plummer dispersion at
/// the subhalo scale radius.  Collisional markers maintain
/// internal_energy=(3/2)*mass*internal_velocity_dispersion^2.
struct WeightedPICMarker {
    double mass = 0.0;
    double r = 0.0;
    double w = 0.0;
    double eta = 0.0;
    std::uint64_t id = 0;
    std::uint64_t source_cell = 0;
    WeightedPICMarkerState state = WeightedPICMarkerState::active;
    double inner_boundary_speed = 0.0;
    double remaining_inner_time = 0.0;
    double internal_energy = 0.0;
    double internal_velocity_dispersion = 0.0;
};

/// Minimal rollback storage for a provisional self-consistent KDK step.
///
/// Collisionless KDK changes only marker radius and radial velocity.  Keeping
/// those two arrays, rather than cloning every marker field, makes endpoint
/// clock probes affordable while preserving marker weights, identities,
/// internal heat, and collision RNG state exactly.
struct WeightedPICOrbitRollbackBuffer {
    std::vector<double> radius;
    std::vector<double> radial_velocity;
};

// A provisional KDK orbit changes radius and radial velocity only. Capture
// those components before a clock trial so rejection can restore the exact input
// without touching weights, internal energies or collision random counters.
inline void capture_weighted_pic_orbit_rollback(
    const std::vector<WeightedPICMarker>& markers,
    WeightedPICOrbitRollbackBuffer& rollback,
    unsigned int requested_threads = std::thread::hardware_concurrency()) {
    if (requested_threads == 0
        || markers.size()
            > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Weighted PIC orbit rollback received invalid inputs.");
    }
    rollback.radius.resize(markers.size());
    rollback.radial_velocity.resize(markers.size());
    const int workers = static_cast<int>(std::min<std::size_t>(
        requested_threads, std::max<std::size_t>(1, markers.size())));
    parallel_for_chunks(
        0,
        static_cast<int>(markers.size()),
        workers,
        [&](int first, int last, int) {
            for (int q = first; q < last; ++q) {
                const WeightedPICMarker& marker =
                    markers[static_cast<std::size_t>(q)];
                rollback.radius[static_cast<std::size_t>(q)] = marker.r;
                rollback.radial_velocity[static_cast<std::size_t>(q)] =
                    marker.w;
            }
        });
}

inline void restore_weighted_pic_orbit_rollback(
    std::vector<WeightedPICMarker>& markers,
    const WeightedPICOrbitRollbackBuffer& rollback,
    unsigned int requested_threads = std::thread::hardware_concurrency()) {
    if (requested_threads == 0
        || rollback.radius.size() != markers.size()
        || rollback.radial_velocity.size() != markers.size()
        || markers.size()
            > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Weighted PIC orbit rollback state does not match the markers.");
    }
    const int workers = static_cast<int>(std::min<std::size_t>(
        requested_threads, std::max<std::size_t>(1, markers.size())));
    parallel_for_chunks(
        0,
        static_cast<int>(markers.size()),
        workers,
        [&](int first, int last, int) {
            for (int q = first; q < last; ++q) {
                WeightedPICMarker& marker =
                    markers[static_cast<std::size_t>(q)];
                marker.r = rollback.radius[static_cast<std::size_t>(q)];
                marker.w =
                    rollback.radial_velocity[static_cast<std::size_t>(q)];
            }
        });
}

struct WeightedPICSamplerState {
    std::uint64_t seed = UINT64_C(0x6a09e667f3bcc909);
    std::uint64_t next_counter = 0;
};

/// Counter state for the optional PIC collision extension.
///
/// Keeping this stream separate from initialization means that checkpointed
/// collision trajectories do not depend on whether an IC was freshly sampled
/// or read from disk.
struct WeightedPICCollisionSamplerState {
    std::uint64_t seed = UINT64_C(0x243f6a8885a308d3);
    std::uint64_t next_counter = 0;
};

enum class WeightedPICVelocityPairing {
    automatic,
    disabled,
    required,
};

struct WeightedPICInitializationOptions {
    int markers_per_occupied_cell = 32;
    std::size_t global_marker_budget = 1000000;
    WeightedPICVelocityPairing velocity_pairing =
        WeightedPICVelocityPairing::automatic;
    double symmetry_tolerance = 256.0 * std::numeric_limits<double>::epsilon();
    int analytic_rejection_max_attempts = 65536;
    // Cell averages do not encode a sharp within-cell support boundary.  The
    // plain initializer therefore requires callers to acknowledge a
    // piecewise-constant reconstruction; analytic equilibria should use the
    // density-aware overload below.
    bool use_piecewise_constant_in_cell = false;
};

struct WeightedPICInitializationDiagnostics {
    std::size_t occupied_source_cells = 0;
    std::size_t marker_count = 0;
    double source_mass = 0.0;
    double represented_mass = 0.0;
    bool velocity_paired = false;
    bool density_aware_in_cell_sampling = false;
};

struct WeightedPICInitializationResult {
    std::vector<WeightedPICMarker> markers;
    WeightedPICInitializationDiagnostics diagnostics;
};

inline std::uint64_t weighted_pic_mix64(std::uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

/// Stateless counter-based variate in the open interval (0,1).
inline double weighted_pic_counter_uniform(
    std::uint64_t seed,
    std::uint64_t counter,
    std::uint64_t dimension) {
    const std::uint64_t word = weighted_pic_mix64(
        seed ^ weighted_pic_mix64(counter)
        ^ weighted_pic_mix64(dimension + UINT64_C(0xd1b54a32d192ed03)));
    const std::uint64_t mantissa = word >> 11;
    const double value = (static_cast<double>(mantissa) + 0.5)
        * (1.0 / 9007199254740992.0);
    // The upper 53-bit midpoint rounds to 1 in binary64. Keep inverse-CDF
    // and rejection samplers inside their declared open unit interval.
    return std::min(value, std::nextafter(1.0, 0.0));
}

inline double weighted_pic_radical_inverse(std::uint64_t index, int base) {
    double inverse = 1.0 / static_cast<double>(base);
    double result = 0.0;
    while (index != 0) {
        result += static_cast<double>(index % static_cast<std::uint64_t>(base))
            * inverse;
        index /= static_cast<std::uint64_t>(base);
        inverse /= static_cast<double>(base);
    }
    return result;
}

inline double weighted_pic_quiet_coordinate(
    std::size_t local_ordinal,
    int base,
    std::uint64_t seed,
    std::uint64_t epoch,
    std::uint64_t source,
    std::uint64_t dimension) {
    const double shift = weighted_pic_counter_uniform(
        seed, epoch ^ weighted_pic_mix64(source), dimension);
    double coordinate = std::fmod(
        weighted_pic_radical_inverse(local_ordinal + 1, base) + shift,
        1.0);
    const double guard = 0.5 * std::numeric_limits<double>::epsilon();
    coordinate = std::clamp(coordinate, guard, 1.0 - guard);
    return coordinate;
}

namespace weighted_pic_detail {

// Neumaier compensation keeps mass ledgers accurate when long double has
// only binary64 precision, as on Apple ARM64.  Retain long-double storage so
// platforms with extended precision also keep their wider accumulator.
struct CompensatedMassSum {
    long double sum = 0.0L;
    long double correction = 0.0L;

    void add(long double mass) {
        const long double updated = sum + mass;
        correction += std::abs(sum) >= std::abs(mass)
            ? (sum - updated) + mass
            : (mass - updated) + sum;
        sum = updated;
    }

    long double value() const { return sum + correction; }
};

}  // namespace weighted_pic_detail

inline double weighted_pic_total_marker_mass(
    const std::vector<WeightedPICMarker>& markers) {
    weighted_pic_detail::CompensatedMassSum mass;
    for (const WeightedPICMarker& marker : markers) {
        mass.add(static_cast<long double>(marker.mass));
    }
    return static_cast<double>(mass.value());
}

inline void validate_weighted_pic_marker(
    const WeightedPICMarker& marker,
    const char* context) {
    if (!(marker.mass > 0.0) || !std::isfinite(marker.mass)
        || !(marker.r >= 0.0) || !std::isfinite(marker.r)
        || !std::isfinite(marker.w)
        || !(marker.eta >= 0.0) || !std::isfinite(marker.eta)
        || !(marker.internal_energy >= 0.0)
        || !std::isfinite(marker.internal_energy)
        || !(marker.internal_velocity_dispersion >= 0.0)
        || !std::isfinite(marker.internal_velocity_dispersion)
        || !(marker.inner_boundary_speed >= 0.0)
        || !std::isfinite(marker.inner_boundary_speed)
        || !(marker.remaining_inner_time >= 0.0)
        || !std::isfinite(marker.remaining_inner_time)) {
        std::ostringstream message;
        message << context << " contains an invalid weighted PIC marker: "
                << "mass=" << marker.mass << ", r=" << marker.r
                << ", w=" << marker.w << ", eta=" << marker.eta
                << ", internal_energy=" << marker.internal_energy
                << ", internal_velocity_dispersion="
                << marker.internal_velocity_dispersion
                << ", state=" << static_cast<int>(marker.state)
                << ", inner_speed=" << marker.inner_boundary_speed
                << ", remaining_inner_time=" << marker.remaining_inner_time;
        throw std::runtime_error(message.str());
    }
    if (marker.state == WeightedPICMarkerState::delayed_inner
        && !(marker.remaining_inner_time > 0.0)) {
        throw std::runtime_error(
            std::string(context)
            + " contains a delayed marker without a positive return time.");
    }
}

namespace weighted_pic_detail {

struct SourceCell {
    std::size_t flat_index = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    long double mass = 0.0L;
};

struct SymmetricSourcePair {
    std::size_t negative_cell = 0;
    std::size_t positive_cell = 0;
    int i = 0;
    int negative_j = 0;
    int positive_j = 0;
    int k = 0;
    long double mass = 0.0L;
};

inline bool close_symmetric(
    long double first,
    long double second,
    double tolerance,
    long double scale_floor = 0.0L) {
    const long double scale = std::max({
        scale_floor, std::abs(first), std::abs(second)});
    if (scale == 0.0L) {
        return true;
    }
    return std::abs(first - second)
        <= static_cast<long double>(tolerance) * scale;
}

template <class Source>
// Divide the total mass CDF into equal strata and use one shared offset.
// This allocates equal-mass particles across occupied source cells while retaining
// the prescribed finite global population.
inline std::vector<std::size_t> systematic_sources(
    const std::vector<Source>& source,
    std::size_t count,
    long double total_mass,
    double offset) {
    std::vector<std::size_t> selected(count, 0);
    std::size_t q = 0;
    long double cumulative = source.front().mass;
    for (std::size_t s = 0; s < count; ++s) {
        const long double target =
            (static_cast<long double>(s) + offset)
            * total_mass / static_cast<long double>(count);
        while (q + 1 < source.size() && target >= cumulative) {
            ++q;
            cumulative += source[q].mass;
        }
        selected[s] = q;
    }
    return selected;
}

inline double inside_cell(double left, double right, double coordinate) {
    const double value = left + coordinate * (right - left);
    return std::clamp(
        value,
        std::nextafter(left, right),
        std::nextafter(right, left));
}

// Choose the nearest of three representable common particle masses. All
// particles keep equal weights, and their total is the closest rounded value
// to the quadrature mass among these neighbouring choices.
inline void correct_marker_mass_sum(
    std::vector<WeightedPICMarker>& markers,
    double target_mass,
    bool paired) {
    (void)paired;
    const double initial = markers.front().mass;
    const std::array<double, 3> candidate{
        initial,
        std::nextafter(initial, 0.0),
        std::nextafter(initial, std::numeric_limits<double>::infinity()),
    };
    double best = initial;
    long double best_error = std::numeric_limits<long double>::infinity();
    for (const double value : candidate) {
        const long double represented = static_cast<long double>(markers.size())
            * static_cast<long double>(value);
        const long double error = std::abs(
            represented - static_cast<long double>(target_mass));
        if (error < best_error) {
            best = value;
            best_error = error;
        }
    }
    for (WeightedPICMarker& marker : markers) {
        marker.mass = best;
    }
}

}  // namespace weighted_pic_detail

/// Samples cell-average f with a fixed global marker budget.
///
/// Systematic mass-CDF strata determine source cells.  Thus a tiny positive
/// tail cell does not receive a compulsory marker, while total represented
/// mass remains exact to roundoff.  Coordinates inside each selected cell use
/// shifted low-discrepancy radical-inverse sequences in canonical dr dw deta.
inline WeightedPICInitializationResult initialize_weighted_vlasov_pic(
    const std::vector<double>& f,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    const WeightedPICInitializationOptions& options,
    WeightedPICSamplerState& sampler) {
    if (p.physics_model != "vlasov_poisson"
        || !using_cell_average_eta_grid(p)) {
        throw std::runtime_error(
            "Weighted PIC initialization requires Vlasov-Poisson and "
            "kinetic_grid=cell_average_eta.");
    }
    validate_cell_average_eta_geometry(geometry, p, "weighted PIC initialization");
    const std::size_t expected = static_cast<std::size_t>(p.Nr) * p.Nw * p.NL;
    validate_finite_nonnegative_distribution(
        f, expected, "weighted PIC initialization");
    if (!options.use_piecewise_constant_in_cell) {
        throw std::runtime_error(
            "Cell averages do not determine within-cell support.  Set "
            "use_piecewise_constant_in_cell=true only for that explicit "
            "reconstruction, or call the analytic-density PIC initializer.");
    }
    if (options.markers_per_occupied_cell < 10
        || options.markers_per_occupied_cell > 1000
        || options.global_marker_budget == 0
        || options.analytic_rejection_max_attempts < 1
        || !(options.symmetry_tolerance >= 0.0)
        || !std::isfinite(options.symmetry_tolerance)) {
        throw std::runtime_error(
            "Weighted PIC initialization requires 10--1000 nominal markers per "
            "occupied cell, a positive global budget, and a finite symmetry tolerance.");
    }

    // Integrate each source cell with 8*pi^2*Delta_r*Delta_w*Delta_eta.
    // Systematic selection then represents this mass measure with the available
    // particles; finite sampling determines which small tail cells receive particles.
    std::vector<weighted_pic_detail::SourceCell> source;
    source.reserve(expected);
    std::vector<long double> mass_by_cell(expected, 0.0L);
    constexpr long double phase_factor =
        8.0L * static_cast<long double>(PI) * static_cast<long double>(PI);
    long double total_mass_ld = 0.0L;
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t flat = idx(i, j, k, p);
                const long double mass = phase_factor
                    * static_cast<long double>(canonical_cell_measure(
                        geometry, i, j, k))
                    * static_cast<long double>(f[flat]);
                mass_by_cell[flat] = mass;
                if (mass > 0.0L) {
                    source.push_back({flat, i, j, k, mass});
                    total_mass_ld += mass;
                }
            }
        }
    }
    if (source.empty() || !(total_mass_ld > 0.0L)
        || !std::isfinite(total_mass_ld)) {
        throw std::runtime_error(
            "Weighted PIC initialization requires a finite positive source mass.");
    }

    const std::size_t occupied = source.size();
    const std::size_t nominal = occupied
        > std::numeric_limits<std::size_t>::max()
                / static_cast<std::size_t>(options.markers_per_occupied_cell)
        ? std::numeric_limits<std::size_t>::max()
        : occupied * static_cast<std::size_t>(options.markers_per_occupied_cell);
    std::size_t marker_count = std::min(nominal, options.global_marker_budget);

    // Opposite radial velocities can be paired when both the velocity faces and
    // source masses have the required symmetry. Paired particles share r, eta and
    // weight so their initial radial current cancels.
    bool symmetric = true;
    for (int j = 0; j <= (p.Nw - 1) / 2 && symmetric; ++j) {
        const int mirror = p.Nw - 1 - j;
        const double left = geometry.radial_velocity.edge[static_cast<std::size_t>(j)];
        const double mirrored_right =
            -geometry.radial_velocity.edge[static_cast<std::size_t>(mirror + 1)];
        if (!weighted_pic_detail::close_symmetric(
                left, mirrored_right, options.symmetry_tolerance, 1.0L)) {
            symmetric = false;
            break;
        }
        for (int i = 0; i < p.Nr && symmetric; ++i) {
            for (int k = 0; k < p.NL; ++k) {
                if (!weighted_pic_detail::close_symmetric(
                        mass_by_cell[idx(i, j, k, p)],
                        mass_by_cell[idx(i, mirror, k, p)],
                        options.symmetry_tolerance)) {
                    symmetric = false;
                    break;
                }
            }
        }
    }
    const bool pair = options.velocity_pairing
            == WeightedPICVelocityPairing::required
        || (options.velocity_pairing == WeightedPICVelocityPairing::automatic
            && symmetric);
    if (options.velocity_pairing == WeightedPICVelocityPairing::required
        && !symmetric) {
        throw std::runtime_error(
            "Weighted PIC +w/-w pairing was required, but source-cell masses "
            "or radial-velocity faces are not symmetric.");
    }
    if (pair) {
        if (marker_count < 2) {
            throw std::runtime_error(
                "Weighted PIC velocity pairing requires a budget of at least two markers.");
        }
        marker_count -= marker_count % 2;
    }
    if (marker_count == 0) {
        throw std::runtime_error("Weighted PIC marker budget selected no markers.");
    }
    if (sampler.next_counter
        > std::numeric_limits<std::uint64_t>::max() - marker_count - 1) {
        throw std::runtime_error("Weighted PIC sampler counter overflow.");
    }

    const std::uint64_t epoch = sampler.next_counter;
    const double cdf_offset = weighted_pic_counter_uniform(
        sampler.seed, epoch, UINT64_C(0x434446));
    const double total_mass = static_cast<double>(total_mass_ld);
    WeightedPICInitializationResult result;
    result.markers.reserve(marker_count);

    if (!pair) {
        const std::vector<std::size_t> selected =
            weighted_pic_detail::systematic_sources(
                source, marker_count, total_mass_ld, cdf_offset);
        std::vector<std::size_t> ordinal(source.size(), 0);
        const double marker_mass = total_mass / static_cast<double>(marker_count);
        for (std::size_t s = 0; s < marker_count; ++s) {
            const std::size_t q = selected[s];
            const weighted_pic_detail::SourceCell& cell = source[q];
            const std::size_t local = ordinal[q]++;
            const double ur = weighted_pic_quiet_coordinate(
                local, 2, sampler.seed, epoch, cell.flat_index, 0);
            const double uw = weighted_pic_quiet_coordinate(
                local, 3, sampler.seed, epoch, cell.flat_index, 1);
            const double ue = weighted_pic_quiet_coordinate(
                local, 5, sampler.seed, epoch, cell.flat_index, 2);
            WeightedPICMarker marker;
            marker.mass = marker_mass;
            marker.r = weighted_pic_detail::inside_cell(
                geometry.radial.edge[static_cast<std::size_t>(cell.i)],
                geometry.radial.edge[static_cast<std::size_t>(cell.i + 1)], ur);
            marker.w = weighted_pic_detail::inside_cell(
                geometry.radial_velocity.edge[static_cast<std::size_t>(cell.j)],
                geometry.radial_velocity.edge[static_cast<std::size_t>(cell.j + 1)], uw);
            marker.eta = weighted_pic_detail::inside_cell(
                geometry.eta.edge[static_cast<std::size_t>(cell.k)],
                geometry.eta.edge[static_cast<std::size_t>(cell.k + 1)], ue);
            marker.id = epoch + s;
            marker.source_cell = cell.flat_index;
            result.markers.push_back(marker);
        }
    } else {
        std::vector<weighted_pic_detail::SymmetricSourcePair> pairs;
        for (int i = 0; i < p.Nr; ++i) {
            for (int j = 0; j <= (p.Nw - 1) / 2; ++j) {
                const int mirror = p.Nw - 1 - j;
                for (int k = 0; k < p.NL; ++k) {
                    const std::size_t negative = idx(i, j, k, p);
                    const std::size_t positive = idx(i, mirror, k, p);
                    const long double pair_mass = j == mirror
                        ? mass_by_cell[negative]
                        : mass_by_cell[negative] + mass_by_cell[positive];
                    if (pair_mass > 0.0L) {
                        pairs.push_back({
                            negative, positive, i, j, mirror, k, pair_mass});
                    }
                }
            }
        }
        const std::size_t pair_count = marker_count / 2;
        const std::vector<std::size_t> selected =
            weighted_pic_detail::systematic_sources(
                pairs, pair_count, total_mass_ld, cdf_offset);
        std::vector<std::size_t> ordinal(pairs.size(), 0);
        const double marker_mass = total_mass / static_cast<double>(marker_count);
        for (std::size_t s = 0; s < pair_count; ++s) {
            const std::size_t q = selected[s];
            const weighted_pic_detail::SymmetricSourcePair& cell = pairs[q];
            const std::size_t local = ordinal[q]++;
            const double ur = weighted_pic_quiet_coordinate(
                local, 2, sampler.seed, epoch, cell.positive_cell, 0);
            const double uw = weighted_pic_quiet_coordinate(
                local, 3, sampler.seed, epoch, cell.positive_cell, 1);
            const double ue = weighted_pic_quiet_coordinate(
                local, 5, sampler.seed, epoch, cell.positive_cell, 2);
            const double radius = weighted_pic_detail::inside_cell(
                geometry.radial.edge[static_cast<std::size_t>(cell.i)],
                geometry.radial.edge[static_cast<std::size_t>(cell.i + 1)], ur);
            const double eta = weighted_pic_detail::inside_cell(
                geometry.eta.edge[static_cast<std::size_t>(cell.k)],
                geometry.eta.edge[static_cast<std::size_t>(cell.k + 1)], ue);
            double positive_w = 0.0;
            if (cell.negative_j == cell.positive_j) {
                positive_w = uw
                    * geometry.radial_velocity.edge[
                        static_cast<std::size_t>(cell.positive_j + 1)];
            } else {
                positive_w = weighted_pic_detail::inside_cell(
                    geometry.radial_velocity.edge[
                        static_cast<std::size_t>(cell.positive_j)],
                    geometry.radial_velocity.edge[
                        static_cast<std::size_t>(cell.positive_j + 1)], uw);
            }
            WeightedPICMarker negative_marker;
            negative_marker.mass = marker_mass;
            negative_marker.r = radius;
            negative_marker.w = -positive_w;
            negative_marker.eta = eta;
            negative_marker.id = epoch + 2 * s;
            negative_marker.source_cell = cell.negative_cell;
            WeightedPICMarker positive_marker = negative_marker;
            positive_marker.w = positive_w;
            positive_marker.id = epoch + 2 * s + 1;
            positive_marker.source_cell = cell.positive_cell;
            result.markers.push_back(negative_marker);
            result.markers.push_back(positive_marker);
        }
    }

    weighted_pic_detail::correct_marker_mass_sum(
        result.markers, total_mass, pair);
    sampler.next_counter += marker_count + 1;
    result.diagnostics.occupied_source_cells = occupied;
    result.diagnostics.marker_count = result.markers.size();
    result.diagnostics.source_mass = total_mass;
    result.diagnostics.represented_mass =
        weighted_pic_total_marker_mass(result.markers);
    result.diagnostics.velocity_paired = pair;
    return result;
}

namespace weighted_pic_detail {

struct CellBounds {
    double r_lower = 0.0;
    double r_upper = 0.0;
    double w_lower = 0.0;
    double w_upper = 0.0;
    double eta_lower = 0.0;
    double eta_upper = 0.0;
};

inline std::array<int, 3> decode_source_cell(
    std::uint64_t source_cell,
    const Params& p) {
    const std::uint64_t velocity_cells =
        static_cast<std::uint64_t>(p.Nw) * p.NL;
    const std::uint64_t expected =
        static_cast<std::uint64_t>(p.Nr) * velocity_cells;
    if (source_cell >= expected) {
        throw std::runtime_error("Weighted PIC marker has an invalid source cell.");
    }
    const int i = static_cast<int>(source_cell / velocity_cells);
    const int remainder = static_cast<int>(source_cell % velocity_cells);
    return {i, remainder / p.NL, remainder % p.NL};
}

inline CellBounds source_cell_bounds(
    std::uint64_t source_cell,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    const std::array<int, 3> cell = decode_source_cell(source_cell, p);
    return {
        geometry.radial.edge[static_cast<std::size_t>(cell[0])],
        geometry.radial.edge[static_cast<std::size_t>(cell[0] + 1)],
        geometry.radial_velocity.edge[static_cast<std::size_t>(cell[1])],
        geometry.radial_velocity.edge[static_cast<std::size_t>(cell[1] + 1)],
        geometry.eta.edge[static_cast<std::size_t>(cell[2])],
        geometry.eta.edge[static_cast<std::size_t>(cell[2] + 1)],
    };
}

template <class AnalyticDensity, class CellDensityUpperBound>
// Condition the sampled location on the actual analytic f inside its source
// cell. A finite positive upper bound controls rejection and catches an invalid
// proposal envelope before accepting a point beyond the physical support.
inline std::array<double, 3> sample_analytic_cell_density(
    const CellBounds& bounds,
    std::uint64_t marker_id,
    const WeightedPICSamplerState& sampler,
    int maximum_attempts,
    AnalyticDensity&& density,
    CellDensityUpperBound&& density_upper_bound) {
    const double upper = density_upper_bound(
        bounds.r_lower,
        bounds.r_upper,
        bounds.w_lower,
        bounds.w_upper,
        bounds.eta_lower,
        bounds.eta_upper);
    if (!(upper > 0.0) || !std::isfinite(upper)) {
        throw std::runtime_error(
            "An occupied weighted PIC source cell has no positive analytic "
            "density upper bound.");
    }
    for (int attempt = 0; attempt < maximum_attempts; ++attempt) {
        const std::uint64_t ordinal = static_cast<std::uint64_t>(attempt);
        const double ur = weighted_pic_quiet_coordinate(
            ordinal, 2, sampler.seed, marker_id, marker_id, 11);
        const double uw = weighted_pic_quiet_coordinate(
            ordinal, 3, sampler.seed, marker_id, marker_id, 12);
        const double ue = weighted_pic_quiet_coordinate(
            ordinal, 5, sampler.seed, marker_id, marker_id, 13);
        const double ua = weighted_pic_quiet_coordinate(
            ordinal, 7, sampler.seed, marker_id, marker_id, 14);
        const double radius = inside_cell(bounds.r_lower, bounds.r_upper, ur);
        const double radial_velocity =
            inside_cell(bounds.w_lower, bounds.w_upper, uw);
        const double eta = inside_cell(bounds.eta_lower, bounds.eta_upper, ue);
        const double value = density(radius, radial_velocity, eta);
        if (!(value >= 0.0) || !std::isfinite(value)) {
            throw std::runtime_error(
                "Weighted PIC analytic density returned a negative or non-finite value.");
        }
        const double bound_tolerance =
            256.0 * std::numeric_limits<double>::epsilon()
            * std::max(upper, value);
        if (value > upper + bound_tolerance) {
            throw std::runtime_error(
                "Weighted PIC analytic density exceeded its declared cell bound.");
        }
        if (value > 0.0 && ua * upper < value) {
            return {radius, radial_velocity, eta};
        }
    }
    throw std::runtime_error(
        "Weighted PIC analytic in-cell rejection sampler exhausted its attempts.");
}

}  // namespace weighted_pic_detail

/// Samples the actual analytic density inside every selected control volume.
///
/// Cell-integrated f still fixes source-cell selection and exact marker mass.
/// The density and rigorous per-cell upper-bound callbacks only condition the
/// continuous (r,w,eta) location.  This is the required path when a sharp
/// energy support cuts through an otherwise occupied eta cell.
template <class AnalyticDensity, class CellDensityUpperBound>
inline WeightedPICInitializationResult
initialize_weighted_vlasov_pic_from_analytic_density(
    const std::vector<double>& cell_average_f,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    const WeightedPICInitializationOptions& options,
    WeightedPICSamplerState& sampler,
    AnalyticDensity&& density,
    CellDensityUpperBound&& density_upper_bound) {
    WeightedPICInitializationOptions allocation_options = options;
    allocation_options.use_piecewise_constant_in_cell = true;
    WeightedPICInitializationResult result = initialize_weighted_vlasov_pic(
        cell_average_f, geometry, p, allocation_options, sampler);

    if (!result.diagnostics.velocity_paired) {
        for (WeightedPICMarker& marker : result.markers) {
            const weighted_pic_detail::CellBounds bounds =
                weighted_pic_detail::source_cell_bounds(
                    marker.source_cell, geometry, p);
            const std::array<double, 3> point =
                weighted_pic_detail::sample_analytic_cell_density(
                    bounds,
                    marker.id,
                    sampler,
                    options.analytic_rejection_max_attempts,
                    density,
                    density_upper_bound);
            marker.r = point[0];
            marker.w = point[1];
            marker.eta = point[2];
        }
    } else {
        for (std::size_t q = 0; q < result.markers.size(); q += 2) {
            WeightedPICMarker& negative = result.markers[q];
            WeightedPICMarker& positive = result.markers[q + 1];
            weighted_pic_detail::CellBounds bounds =
                weighted_pic_detail::source_cell_bounds(
                    positive.source_cell, geometry, p);
            const std::array<int, 3> cell =
                weighted_pic_detail::decode_source_cell(
                    positive.source_cell, p);
            if (cell[1] == p.Nw - 1 - cell[1]) {
                bounds.w_lower = 0.0;
            }
            const std::array<double, 3> point =
                weighted_pic_detail::sample_analytic_cell_density(
                    bounds,
                    positive.id,
                    sampler,
                    options.analytic_rejection_max_attempts,
                    density,
                    density_upper_bound);
            const double mirrored_density = density(
                point[0], -point[1], point[2]);
            const double sampled_density = density(point[0], point[1], point[2]);
            if (!weighted_pic_detail::close_symmetric(
                    sampled_density,
                    mirrored_density,
                    options.symmetry_tolerance)) {
                throw std::runtime_error(
                    "Weighted PIC analytic density is not pointwise symmetric "
                    "under the requested +w/-w pairing.");
            }
            negative.r = point[0];
            negative.w = -point[1];
            negative.eta = point[2];
            positive.r = point[0];
            positive.w = point[1];
            positive.eta = point[2];
        }
    }
    result.diagnostics.density_aware_in_cell_sampling = true;
    return result;
}

/// Radial finite-volume mesh used by particle deposition.
struct WeightedPICRadialMesh {
    std::vector<double> edge;
    std::vector<double> volume_coordinate_center;
    std::vector<double> shell_volume;
    std::vector<double> venus_center;
    std::vector<double> venus_width;
};

// Use q=r^3/3 for the particle cloud coordinate. Physical shell volume is
// 4*pi*Delta_q, while the phase-grid radial widths remain Delta_r.
inline WeightedPICRadialMesh make_weighted_pic_radial_mesh(
    const PhaseSpaceCellGeometry& geometry) {
    WeightedPICRadialMesh mesh;
    mesh.edge = geometry.radial.edge;
    mesh.venus_center = geometry.radial.center;
    mesh.venus_width = geometry.radial.width;
    const std::size_t cells = geometry.radial.center.size();
    mesh.volume_coordinate_center.resize(cells, 0.0);
    mesh.shell_volume.resize(cells, 0.0);
    for (std::size_t i = 0; i < cells; ++i) {
        const double q0 = mesh.edge[i] * mesh.edge[i] * mesh.edge[i] / 3.0;
        const double q1 = mesh.edge[i + 1] * mesh.edge[i + 1]
            * mesh.edge[i + 1] / 3.0;
        if (!(q1 > q0)) {
            throw std::runtime_error(
                "Weighted PIC radial faces must be positive and increasing.");
        }
        mesh.volume_coordinate_center[i] = 0.5 * (q0 + q1);
        mesh.shell_volume[i] = 4.0 * PI * (q1 - q0);
    }
    return mesh;
}

/// PIC field/deposition grid that may be radially finer than the source
/// phase-space grid.  Refinement changes neither marker weights nor source
/// cells and introduces no post-deposition smoothing.
struct WeightedPICFieldGrid {
    Params parameters;
    WeightedPICRadialMesh mesh;
    std::vector<double> radial_coordinate;
    std::vector<double> radius;
};

inline WeightedPICFieldGrid make_weighted_pic_field_grid(
    const Params& source_parameters,
    int radial_cells) {
    if (source_parameters.physics_model != "vlasov_poisson"
        || !using_cell_average_eta_grid(source_parameters)
        || radial_cells < source_parameters.Nr) {
        throw std::runtime_error(
            "Weighted PIC field grid requires cell-average "
            "Vlasov-Poisson and may not be coarser than the source grid.");
    }
    WeightedPICFieldGrid result;
    result.parameters = source_parameters;
    result.parameters.Nr = radial_cells;
    const PhaseSpaceCellGeometry geometry = make_cell_average_eta_geometry(
        make_positive_radial_cell_axis(result.parameters), result.parameters);
    result.mesh = make_weighted_pic_radial_mesh(geometry);
    result.radius = geometry.radial.center;
    result.radial_coordinate =
        radial_coordinate_grid(result.radius, result.parameters);
    return result;
}

struct WeightedPICCICStencil {
    std::array<std::size_t, 2> cell{0, 0};
    std::array<double, 2> weight{1.0, 0.0};
    int count = 1;
};

enum class WeightedPICParticleShape {
    cic,
    tsc,
};

struct WeightedPICTSCStencil {
    std::array<std::size_t, 3> cell{0, 0, 0};
    std::array<double, 3> weight{1.0, 0.0, 0.0};
    int count = 1;
};

/// A resolved PIC stencil plus the fraction of the same particle cloud that
/// lies inside the unresolved inner sphere.  The two sets of weights form one
/// nonnegative partition of unity.
template <class ResolvedStencil>
struct WeightedPICInnerCarrierStencil {
    ResolvedStencil resolved;
    double hidden_weight = 0.0;
};

/// Compact CIC in q=r^3/3.  At a physical boundary the truncated cloud is
/// renormalized over the in-domain centers, so no marker mass leaks to ghosts.
// Linear cloud weights interpolate between q centres. At a closed boundary,
// the retained nonnegative weights sum to one so the whole mass remains in-domain.
inline WeightedPICCICStencil weighted_pic_cic_stencil(
    double radius,
    const WeightedPICRadialMesh& mesh) {
    if (mesh.edge.size() != mesh.volume_coordinate_center.size() + 1
        || mesh.shell_volume.size() != mesh.volume_coordinate_center.size()
        || mesh.volume_coordinate_center.empty()) {
        throw std::runtime_error("Weighted PIC CIC received an invalid radial mesh.");
    }
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
        * std::max({1.0, std::abs(mesh.edge.front()), std::abs(mesh.edge.back())});
    if (radius < mesh.edge.front() - tolerance
        || radius > mesh.edge.back() + tolerance
        || !std::isfinite(radius)) {
        throw std::runtime_error("Active weighted PIC marker lies outside the radial mesh.");
    }
    radius = std::clamp(radius, mesh.edge.front(), mesh.edge.back());
    const double q = radius * radius * radius / 3.0;
    const std::vector<double>& center = mesh.volume_coordinate_center;
    if (center.size() == 1 || q <= center.front()) {
        return {};
    }
    if (q >= center.back()) {
        WeightedPICCICStencil stencil;
        stencil.cell[0] = center.size() - 1;
        return stencil;
    }
    const auto upper = std::upper_bound(center.begin(), center.end(), q);
    const std::size_t right = static_cast<std::size_t>(upper - center.begin());
    const std::size_t left = right - 1;
    const double fraction = (q - center[left]) / (center[right] - center[left]);
    WeightedPICCICStencil stencil;
    stencil.cell = {left, right};
    stencil.weight = {1.0 - fraction, fraction};
    stencil.count = 2;
    const double norm = stencil.weight[0] + stencil.weight[1];
    stencil.weight[0] /= norm;
    stencil.weight[1] /= norm;
    return stencil;
}

/// Quadratic TSC in the logical coordinate through the nonuniform q centers.
/// The pointwise B-spline is multiplied by the receiving shell volume before
/// normalization.  This finite-volume (Shepard) correction is essential on
/// linear-r grids, whose q=r^3/3 cell volumes vary rapidly near the origin:
/// without it, a smooth core is spuriously concentrated into the first one or
/// two cells.  Out-of-domain cloud pieces are discarded and the surviving
/// volume-weighted cloud remains a nonnegative partition of unity.
// Quadratic logical-coordinate weights are multiplied by receiving shell
// volumes before normalization. This finite-volume correction accounts for rapidly
// changing q-cell sizes near a regular centre.
inline WeightedPICTSCStencil weighted_pic_tsc_stencil(
    double radius,
    const WeightedPICRadialMesh& mesh) {
    if (mesh.edge.size() != mesh.volume_coordinate_center.size() + 1
        || mesh.shell_volume.size() != mesh.volume_coordinate_center.size()
        || mesh.volume_coordinate_center.empty()) {
        throw std::runtime_error("Weighted PIC TSC received an invalid radial mesh.");
    }
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
        * std::max({1.0, std::abs(mesh.edge.front()), std::abs(mesh.edge.back())});
    if (radius < mesh.edge.front() - tolerance
        || radius > mesh.edge.back() + tolerance
        || !std::isfinite(radius)) {
        throw std::runtime_error("Active weighted PIC marker lies outside the radial mesh.");
    }
    radius = std::clamp(radius, mesh.edge.front(), mesh.edge.back());
    const std::vector<double>& center = mesh.volume_coordinate_center;
    if (center.size() == 1) {
        return {};
    }
    const double q = radius * radius * radius / 3.0;
    double logical = 0.0;
    if (q <= center.front()) {
        logical = (q - center.front()) / (center[1] - center[0]);
    } else if (q >= center.back()) {
        logical = static_cast<double>(center.size() - 1)
            + (q - center.back())
                / (center.back() - center[center.size() - 2]);
    } else {
        const auto upper = std::upper_bound(center.begin(), center.end(), q);
        const std::size_t right = static_cast<std::size_t>(upper - center.begin());
        const std::size_t left = right - 1;
        logical = static_cast<double>(left)
            + (q - center[left]) / (center[right] - center[left]);
    }

    const long long nearest = static_cast<long long>(
        std::floor(logical + 0.5));
    const double displacement = logical - static_cast<double>(nearest);
    const std::array<double, 3> candidate_weight{
        0.5 * (0.5 - displacement) * (0.5 - displacement),
        0.75 - displacement * displacement,
        0.5 * (0.5 + displacement) * (0.5 + displacement),
    };
    WeightedPICTSCStencil stencil;
    stencil.count = 0;
    double norm = 0.0;
    for (int offset = -1; offset <= 1; ++offset) {
        const long long index = nearest + offset;
        if (index < 0 || index >= static_cast<long long>(center.size())) {
            continue;
        }
        const double weight =
            candidate_weight[static_cast<std::size_t>(offset + 1)]
            * mesh.shell_volume[static_cast<std::size_t>(index)];
        if (!(weight > 0.0)) {
            continue;
        }
        const std::size_t slot = static_cast<std::size_t>(stencil.count++);
        stencil.cell[slot] = static_cast<std::size_t>(index);
        stencil.weight[slot] = weight;
        norm += weight;
    }
    if (!(norm > 0.0)) {
        throw std::runtime_error("Weighted PIC TSC lost its complete boundary cloud.");
    }
    for (int n = 0; n < stencil.count; ++n) {
        stencil.weight[static_cast<std::size_t>(n)] /= norm;
    }
    return stencil;
}

/// CIC cloud with a continuous unresolved-inner-carrier target.
///
/// The ordinary boundary stencil truncates and renormalizes a cloud because a
/// reflecting or closed mesh has no physical mass beyond its face.  The
/// origin-parity expanding runner is different: active trajectories really
/// pass through 0 <= r < r_min, and that interior mass must source gravity.
/// Immediately below r_min the boundary cloud is therefore handed smoothly
/// to the unresolved central carrier.  The resolved fraction is a cubic
/// smoothstep in the spherical volume coordinate q=r^3/3; it and its first
/// derivative match at both ends of the handoff.  Thus crossing the numerical
/// face cannot move one whole unequal-mass marker discontinuously.
inline WeightedPICInnerCarrierStencil<WeightedPICCICStencil>
weighted_pic_cic_stencil_with_inner_carrier(
    double radius,
    const WeightedPICRadialMesh& mesh) {
    if (mesh.edge.size() != mesh.volume_coordinate_center.size() + 1
        || mesh.shell_volume.size() != mesh.volume_coordinate_center.size()
        || mesh.volume_coordinate_center.empty()) {
        throw std::runtime_error(
            "Weighted PIC inner-carrier CIC received an invalid radial mesh.");
    }
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
        * std::max({1.0, std::abs(mesh.edge.front()), std::abs(mesh.edge.back())});
    if (radius < -tolerance || radius > mesh.edge.back() + tolerance
        || !std::isfinite(radius)) {
        throw std::runtime_error(
            "Origin-parity weighted PIC marker lies outside its radial domain.");
    }
    radius = std::clamp(radius, 0.0, mesh.edge.back());
    const double q = radius * radius * radius / 3.0;
    const double inner_face_q =
        mesh.edge.front() * mesh.edge.front() * mesh.edge.front() / 3.0;
    if (q >= inner_face_q) {
        WeightedPICInnerCarrierStencil<WeightedPICCICStencil> result;
        result.resolved = weighted_pic_cic_stencil(radius, mesh);
        return result;
    }

    const double first_shell_width =
        2.0 * (mesh.volume_coordinate_center.front() - inner_face_q);
    const double transition_width =
        std::min(first_shell_width, inner_face_q);
    if (!(transition_width > 0.0)) {
        throw std::runtime_error(
            "Weighted PIC inner-carrier CIC has no positive handoff width.");
    }
    const long double s = std::clamp(
        (static_cast<long double>(q)
         - static_cast<long double>(inner_face_q - transition_width))
            / static_cast<long double>(transition_width),
        0.0L,
        1.0L);
    const long double resolved_fraction = s * s * (3.0L - 2.0L * s);
    WeightedPICInnerCarrierStencil<WeightedPICCICStencil> result;
    result.hidden_weight = static_cast<double>(1.0L - resolved_fraction);
    result.resolved = weighted_pic_cic_stencil(mesh.edge.front(), mesh);
    for (int n = 0; n < result.resolved.count; ++n) {
        result.resolved.weight[static_cast<std::size_t>(n)] *=
            static_cast<double>(resolved_fraction);
    }
    if (!(resolved_fraction > 0.0L)) {
        result.resolved.count = 0;
    }
    return result;
}

/// TSC cloud with the same smooth unresolved-inner-carrier handoff as CIC.
/// The ordinary, volume-corrected boundary stencil is frozen at r_min below
/// the face and only its total resolved fraction changes.  The complementary
/// fraction is added to the hidden central mass.  Resolved plus hidden weights
/// remain a nonnegative partition of unity at every radius.
inline WeightedPICInnerCarrierStencil<WeightedPICTSCStencil>
weighted_pic_tsc_stencil_with_inner_carrier(
    double radius,
    const WeightedPICRadialMesh& mesh) {
    if (mesh.edge.size() != mesh.volume_coordinate_center.size() + 1
        || mesh.shell_volume.size() != mesh.volume_coordinate_center.size()
        || mesh.volume_coordinate_center.empty()) {
        throw std::runtime_error(
            "Weighted PIC inner-carrier TSC received an invalid radial mesh.");
    }
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
        * std::max({1.0, std::abs(mesh.edge.front()), std::abs(mesh.edge.back())});
    if (radius < -tolerance || radius > mesh.edge.back() + tolerance
        || !std::isfinite(radius)) {
        throw std::runtime_error(
            "Origin-parity weighted PIC marker lies outside its radial domain.");
    }
    radius = std::clamp(radius, 0.0, mesh.edge.back());
    const double q = radius * radius * radius / 3.0;
    const double inner_face_q =
        mesh.edge.front() * mesh.edge.front() * mesh.edge.front() / 3.0;
    if (q >= inner_face_q) {
        WeightedPICInnerCarrierStencil<WeightedPICTSCStencil> result;
        result.resolved = weighted_pic_tsc_stencil(radius, mesh);
        return result;
    }

    const double first_shell_width =
        2.0 * (mesh.volume_coordinate_center.front() - inner_face_q);
    const double transition_width =
        std::min(first_shell_width, inner_face_q);
    if (!(transition_width > 0.0)) {
        throw std::runtime_error(
            "Weighted PIC inner-carrier TSC has no positive handoff width.");
    }
    const long double s = std::clamp(
        (static_cast<long double>(q)
         - static_cast<long double>(inner_face_q - transition_width))
            / static_cast<long double>(transition_width),
        0.0L,
        1.0L);
    const long double resolved_fraction = s * s * (3.0L - 2.0L * s);
    WeightedPICInnerCarrierStencil<WeightedPICTSCStencil> result;
    result.hidden_weight = static_cast<double>(1.0L - resolved_fraction);
    result.resolved = weighted_pic_tsc_stencil(mesh.edge.front(), mesh);
    for (int n = 0; n < result.resolved.count; ++n) {
        result.resolved.weight[static_cast<std::size_t>(n)] *=
            static_cast<double>(resolved_fraction);
    }
    if (!(resolved_fraction > 0.0L)) {
        result.resolved.count = 0;
    }
    return result;
}

// Store extensive shell moments alongside densities. Radial energy flux
// uses w*(w^2+v_t^2)/2 in the peculiar frame; physical interpretation in an
// expanding background also requires the specified observer velocity.
struct WeightedPICDepositedMoments {
    std::vector<double> shell_mass;
    std::vector<double> shell_radial_momentum;
    std::vector<double> shell_radial_second_moment;
    std::vector<double> shell_tangential_second_moment;
    std::vector<double> shell_radial_energy_flux;
    std::vector<double> density;
    std::vector<double> radial_momentum_density;
    std::vector<double> radial_second_moment_density;
    std::vector<double> tangential_second_moment_density;
    std::vector<double> radial_energy_flux_density;
    std::vector<double> effective_marker_number;
    double active_marker_mass = 0.0;
    double deposited_mesh_mass = 0.0;
    double hidden_inner_mass = 0.0;
    double escaped_mass = 0.0;
    double mass_roundoff = 0.0;
};

namespace weighted_pic_detail {

template <class Selector>
// Apply one common particle selector and mass scale to every deposited
// moment. Delayed and escaped masses enter distinct ledgers while active clouds
// contribute density, current, both velocity moments and kinetic energy flux.
inline WeightedPICDepositedMoments deposit_selected(
    const std::vector<WeightedPICMarker>& markers,
    const WeightedPICRadialMesh& mesh,
    Selector&& select,
    double mass_scale,
    WeightedPICParticleShape shape) {
    if (!(mass_scale > 0.0) || !std::isfinite(mass_scale)) {
        throw std::runtime_error("Weighted PIC deposition received an invalid mass scale.");
    }
    const std::size_t cells = mesh.shell_volume.size();
    if (cells == 0 || mesh.volume_coordinate_center.size() != cells
        || mesh.venus_center.size() != cells || mesh.venus_width.size() != cells) {
        throw std::runtime_error("Weighted PIC deposition received an invalid radial mesh.");
    }
    std::vector<long double> mass(cells, 0.0L);
    std::vector<long double> momentum(cells, 0.0L);
    std::vector<long double> radial_second(cells, 0.0L);
    std::vector<long double> tangential_second(cells, 0.0L);
    std::vector<long double> energy_flux(cells, 0.0L);
    std::vector<long double> mass_square(cells, 0.0L);
    CompensatedMassSum active_mass;
    CompensatedMassSum hidden_mass;
    CompensatedMassSum escaped_mass;
    for (const WeightedPICMarker& marker : markers) {
        validate_weighted_pic_marker(marker, "Weighted PIC deposition");
        if (!select(marker)) {
            continue;
        }
        const long double scaled_mass =
            static_cast<long double>(mass_scale) * marker.mass;
        if (marker.state == WeightedPICMarkerState::delayed_inner) {
            hidden_mass.add(scaled_mass);
            continue;
        }
        if (marker.state == WeightedPICMarkerState::escaped) {
            escaped_mass.add(scaled_mass);
            continue;
        }
        active_mass.add(scaled_mass);
        const long double vt2 = 2.0L * marker.eta
            / std::max(
                static_cast<long double>(marker.r) * marker.r,
                1.0e-300L);
        const long double w = marker.w;
        const long double specific_energy_flux =
            0.5L * w * (w * w + vt2);
        auto deposit_one = [&](std::size_t i, double shape_weight) {
            const long double contribution = scaled_mass * shape_weight;
            mass[i] += contribution;
            momentum[i] += contribution * w;
            radial_second[i] += contribution * w * w;
            tangential_second[i] += contribution * vt2;
            energy_flux[i] += contribution * specific_energy_flux;
            mass_square[i] += contribution * contribution;
        };
        if (shape == WeightedPICParticleShape::cic) {
            const WeightedPICCICStencil stencil =
                weighted_pic_cic_stencil(marker.r, mesh);
            for (int n = 0; n < stencil.count; ++n) {
                deposit_one(
                    stencil.cell[static_cast<std::size_t>(n)],
                    stencil.weight[static_cast<std::size_t>(n)]);
            }
        } else {
            const WeightedPICTSCStencil stencil =
                weighted_pic_tsc_stencil(marker.r, mesh);
            for (int n = 0; n < stencil.count; ++n) {
                deposit_one(
                    stencil.cell[static_cast<std::size_t>(n)],
                    stencil.weight[static_cast<std::size_t>(n)]);
            }
        }
    }

    WeightedPICDepositedMoments result;
    result.shell_mass.resize(cells, 0.0);
    result.shell_radial_momentum.resize(cells, 0.0);
    result.shell_radial_second_moment.resize(cells, 0.0);
    result.shell_tangential_second_moment.resize(cells, 0.0);
    result.shell_radial_energy_flux.resize(cells, 0.0);
    result.density.resize(cells, 0.0);
    result.radial_momentum_density.resize(cells, 0.0);
    result.radial_second_moment_density.resize(cells, 0.0);
    result.tangential_second_moment_density.resize(cells, 0.0);
    result.radial_energy_flux_density.resize(cells, 0.0);
    result.effective_marker_number.resize(cells, 0.0);
    CompensatedMassSum deposited;
    for (std::size_t i = 0; i < cells; ++i) {
        result.shell_mass[i] = static_cast<double>(mass[i]);
        result.shell_radial_momentum[i] = static_cast<double>(momentum[i]);
        result.shell_radial_second_moment[i] =
            static_cast<double>(radial_second[i]);
        result.shell_tangential_second_moment[i] =
            static_cast<double>(tangential_second[i]);
        result.shell_radial_energy_flux[i] =
            static_cast<double>(energy_flux[i]);
        const double inverse_volume = 1.0 / mesh.shell_volume[i];
        result.density[i] = result.shell_mass[i] * inverse_volume;
        result.radial_momentum_density[i] =
            result.shell_radial_momentum[i] * inverse_volume;
        result.radial_second_moment_density[i] =
            result.shell_radial_second_moment[i] * inverse_volume;
        result.tangential_second_moment_density[i] =
            result.shell_tangential_second_moment[i] * inverse_volume;
        result.radial_energy_flux_density[i] =
            result.shell_radial_energy_flux[i] * inverse_volume;
        result.effective_marker_number[i] = mass_square[i] > 0.0L
            ? static_cast<double>(mass[i] * mass[i] / mass_square[i])
            : 0.0;
        deposited.add(mass[i]);
    }
    result.active_marker_mass = static_cast<double>(active_mass.value());
    result.deposited_mesh_mass = static_cast<double>(deposited.value());
    result.hidden_inner_mass = static_cast<double>(hidden_mass.value());
    result.escaped_mass = static_cast<double>(escaped_mass.value());
    result.mass_roundoff = static_cast<double>(
        deposited.value() - active_mass.value());
    return result;
}

}  // namespace weighted_pic_detail

/// Deposits all active markers with conservative spherical CIC.
inline WeightedPICDepositedMoments deposit_weighted_vlasov_pic(
    const std::vector<WeightedPICMarker>& markers,
    const WeightedPICRadialMesh& mesh,
    WeightedPICParticleShape shape = WeightedPICParticleShape::cic) {
    return weighted_pic_detail::deposit_selected(
        markers, mesh, [](const WeightedPICMarker&) { return true; }, 1.0, shape);
}

/// Thread-parallel full-marker deposition with deterministic thread-order
/// reduction.  Each worker owns a private radial accumulator, so no atomics are
/// used in the marker loop.  This is the production path for large PIC runs;
/// the serial routine above remains useful when bitwise thread-count
/// independence is required.
inline WeightedPICDepositedMoments deposit_weighted_vlasov_pic_parallel(
    const std::vector<WeightedPICMarker>& markers,
    const WeightedPICRadialMesh& mesh,
    WeightedPICParticleShape shape,
    unsigned int requested_threads = std::thread::hardware_concurrency(),
    bool carry_active_mass_inside_inner_face = false) {
    const std::size_t cells = mesh.shell_volume.size();
    if (cells == 0 || mesh.volume_coordinate_center.size() != cells
        || mesh.venus_center.size() != cells || mesh.venus_width.size() != cells
        || requested_threads == 0
        || markers.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Parallel weighted PIC deposition received invalid inputs.");
    }
    const int workers = static_cast<int>(std::min<std::size_t>(
        requested_threads, std::max<std::size_t>(1, markers.size())));
    struct Partial {
        std::vector<long double> mass;
        std::vector<long double> momentum;
        std::vector<long double> radial_second;
        std::vector<long double> tangential_second;
        std::vector<long double> energy_flux;
        std::vector<long double> mass_square;
        weighted_pic_detail::CompensatedMassSum active_mass;
        weighted_pic_detail::CompensatedMassSum hidden_mass;
        weighted_pic_detail::CompensatedMassSum escaped_mass;
    };
    std::vector<Partial> partial(static_cast<std::size_t>(workers));
    for (Partial& value : partial) {
        value.mass.assign(cells, 0.0L);
        value.momentum.assign(cells, 0.0L);
        value.radial_second.assign(cells, 0.0L);
        value.tangential_second.assign(cells, 0.0L);
        value.energy_flux.assign(cells, 0.0L);
        value.mass_square.assign(cells, 0.0L);
    }
    parallel_for_chunks(
        0,
        static_cast<int>(markers.size()),
        workers,
        [&](int first, int last, int thread) {
            Partial& local = partial[static_cast<std::size_t>(thread)];
            for (int q = first; q < last; ++q) {
                const WeightedPICMarker& marker =
                    markers[static_cast<std::size_t>(q)];
                validate_weighted_pic_marker(
                    marker, "Parallel weighted PIC deposition");
                const long double marker_mass = marker.mass;
                if (marker.state == WeightedPICMarkerState::delayed_inner) {
                    local.hidden_mass.add(marker_mass);
                    continue;
                }
                if (marker.state == WeightedPICMarkerState::escaped) {
                    local.escaped_mass.add(marker_mass);
                    continue;
                }
                const long double w = marker.w;
                const long double vt2 = 2.0L * marker.eta
                    / std::max(
                        static_cast<long double>(marker.r) * marker.r,
                        1.0e-300L);
                const long double specific_energy_flux =
                    0.5L * w * (w * w + vt2);
                auto deposit_one = [&](std::size_t cell, double weight) {
                    const long double contribution = marker_mass * weight;
                    local.mass[cell] += contribution;
                    local.momentum[cell] += contribution * w;
                    local.radial_second[cell] += contribution * w * w;
                    local.tangential_second[cell] += contribution * vt2;
                    local.energy_flux[cell] +=
                        contribution * specific_energy_flux;
                    local.mass_square[cell] += contribution * contribution;
                };
                if (shape == WeightedPICParticleShape::cic) {
                    if (carry_active_mass_inside_inner_face) {
                        // An origin-parity marker remains dynamical below the
                        // first field face.  Partition its finite cloud
                        // continuously between the resolved mesh and the
                        // unresolved central gravity carrier; never switch an
                        // entire (possibly heavy) marker at one comparison.
                        const auto stencil =
                            weighted_pic_cic_stencil_with_inner_carrier(
                                marker.r, mesh);
                        long double resolved_weight = 0.0L;
                        for (int n = 0; n < stencil.resolved.count; ++n) {
                            const double weight = stencil.resolved.weight[
                                static_cast<std::size_t>(n)];
                            resolved_weight += weight;
                            deposit_one(
                                stencil.resolved.cell[
                                    static_cast<std::size_t>(n)],
                                weight);
                        }
                        local.active_mass.add(marker_mass * resolved_weight);
                        local.hidden_mass.add(marker_mass
                            * static_cast<long double>(
                                stencil.hidden_weight));
                    } else {
                        const WeightedPICCICStencil stencil =
                            weighted_pic_cic_stencil(marker.r, mesh);
                        local.active_mass.add(marker_mass);
                        for (int n = 0; n < stencil.count; ++n) {
                            deposit_one(
                                stencil.cell[static_cast<std::size_t>(n)],
                                stencil.weight[static_cast<std::size_t>(n)]);
                        }
                    }
                } else {
                    if (carry_active_mass_inside_inner_face) {
                        const auto stencil =
                            weighted_pic_tsc_stencil_with_inner_carrier(
                                marker.r, mesh);
                        long double resolved_weight = 0.0L;
                        for (int n = 0; n < stencil.resolved.count; ++n) {
                            const double weight = stencil.resolved.weight[
                                static_cast<std::size_t>(n)];
                            resolved_weight += weight;
                            deposit_one(
                                stencil.resolved.cell[
                                    static_cast<std::size_t>(n)],
                                weight);
                        }
                        local.active_mass.add(marker_mass * resolved_weight);
                        local.hidden_mass.add(marker_mass
                            * static_cast<long double>(
                                stencil.hidden_weight));
                    } else {
                        const WeightedPICTSCStencil stencil =
                            weighted_pic_tsc_stencil(marker.r, mesh);
                        local.active_mass.add(marker_mass);
                        for (int n = 0; n < stencil.count; ++n) {
                            deposit_one(
                                stencil.cell[static_cast<std::size_t>(n)],
                                stencil.weight[static_cast<std::size_t>(n)]);
                        }
                    }
                }
            }
        });

    WeightedPICDepositedMoments result;
    result.shell_mass.assign(cells, 0.0);
    result.shell_radial_momentum.assign(cells, 0.0);
    result.shell_radial_second_moment.assign(cells, 0.0);
    result.shell_tangential_second_moment.assign(cells, 0.0);
    result.shell_radial_energy_flux.assign(cells, 0.0);
    result.density.assign(cells, 0.0);
    result.radial_momentum_density.assign(cells, 0.0);
    result.radial_second_moment_density.assign(cells, 0.0);
    result.tangential_second_moment_density.assign(cells, 0.0);
    result.radial_energy_flux_density.assign(cells, 0.0);
    result.effective_marker_number.assign(cells, 0.0);
    weighted_pic_detail::CompensatedMassSum deposited;
    weighted_pic_detail::CompensatedMassSum active;
    weighted_pic_detail::CompensatedMassSum hidden;
    weighted_pic_detail::CompensatedMassSum escaped;
    // Reduce disjoint worker accumulators in a fixed order. The same deposited
    // mass closes the resolved/hidden/escaped budget and normalizes N_eff.
    for (const Partial& local : partial) {
        active.add(local.active_mass.value());
        hidden.add(local.hidden_mass.value());
        escaped.add(local.escaped_mass.value());
    }
    for (std::size_t cell = 0; cell < cells; ++cell) {
        long double mass = 0.0L;
        long double momentum = 0.0L;
        long double radial_second = 0.0L;
        long double tangential_second = 0.0L;
        long double energy_flux = 0.0L;
        long double mass_square = 0.0L;
        for (const Partial& local : partial) {
            mass += local.mass[cell];
            momentum += local.momentum[cell];
            radial_second += local.radial_second[cell];
            tangential_second += local.tangential_second[cell];
            energy_flux += local.energy_flux[cell];
            mass_square += local.mass_square[cell];
        }
        result.shell_mass[cell] = static_cast<double>(mass);
        result.shell_radial_momentum[cell] = static_cast<double>(momentum);
        result.shell_radial_second_moment[cell] =
            static_cast<double>(radial_second);
        result.shell_tangential_second_moment[cell] =
            static_cast<double>(tangential_second);
        result.shell_radial_energy_flux[cell] = static_cast<double>(energy_flux);
        const double inverse_volume = 1.0 / mesh.shell_volume[cell];
        result.density[cell] = result.shell_mass[cell] * inverse_volume;
        result.radial_momentum_density[cell] =
            result.shell_radial_momentum[cell] * inverse_volume;
        result.radial_second_moment_density[cell] =
            result.shell_radial_second_moment[cell] * inverse_volume;
        result.tangential_second_moment_density[cell] =
            result.shell_tangential_second_moment[cell] * inverse_volume;
        result.radial_energy_flux_density[cell] =
            result.shell_radial_energy_flux[cell] * inverse_volume;
        result.effective_marker_number[cell] = mass_square > 0.0L
            ? static_cast<double>(mass * mass / mass_square)
            : 0.0;
        deposited.add(mass);
    }
    result.active_marker_mass = static_cast<double>(active.value());
    result.deposited_mesh_mass = static_cast<double>(deposited.value());
    result.hidden_inner_mass = static_cast<double>(hidden.value());
    result.escaped_mass = static_cast<double>(escaped.value());
    result.mass_roundoff = static_cast<double>(
        deposited.value() - active.value());
    return result;
}

struct WeightedPICNoiseDiagnostics {
    std::vector<double> effective_marker_number;
    std::vector<double> density_uncertainty;
    std::vector<double> radial_current_uncertainty;
    std::vector<double> kinetic_energy_density_uncertainty;
    std::vector<double> radial_energy_flux_uncertainty;
    std::size_t batch_a_markers = 0;
    std::size_t batch_b_markers = 0;
    double batch_a_mass = 0.0;
    double batch_b_mass = 0.0;
};

/// Deterministically partitions markers into two independent hash batches.
/// Each batch receives its inverse inclusion-probability factor of two; half
/// their difference is the two-replicate standard error estimate for the full
/// deposition.
// Split identities deterministically into two hash samples and double each
// sample weight. Half their deposited difference estimates finite-sample noise
// without changing the distribution that actually sources the field.
inline WeightedPICNoiseDiagnostics diagnose_weighted_pic_noise(
    const std::vector<WeightedPICMarker>& markers,
    const WeightedPICRadialMesh& mesh,
    std::uint64_t split_seed = UINT64_C(0xbb67ae8584caa73b),
    WeightedPICParticleShape shape = WeightedPICParticleShape::cic) {
    std::array<long double, 2> batch_mass{0.0L, 0.0L};
    std::array<std::size_t, 2> batch_count{0, 0};
    auto batch_of = [&](const WeightedPICMarker& marker) {
        return static_cast<int>(
            weighted_pic_mix64(marker.id ^ split_seed) & UINT64_C(1));
    };
    for (const WeightedPICMarker& marker : markers) {
        validate_weighted_pic_marker(marker, "Weighted PIC noise diagnostics");
        if (marker.state != WeightedPICMarkerState::active) {
            continue;
        }
        const int batch = batch_of(marker);
        batch_mass[static_cast<std::size_t>(batch)] += marker.mass;
        ++batch_count[static_cast<std::size_t>(batch)];
    }
    if (!(batch_mass[0] > 0.0L) || !(batch_mass[1] > 0.0L)) {
        throw std::runtime_error(
            "Weighted PIC noise diagnostics require active markers in both batches.");
    }
    const WeightedPICDepositedMoments full =
        deposit_weighted_vlasov_pic(markers, mesh, shape);
    const WeightedPICDepositedMoments first =
        weighted_pic_detail::deposit_selected(
            markers,
            mesh,
            [&](const WeightedPICMarker& marker) {
                return batch_of(marker) == 0;
            },
            2.0,
            shape);
    const WeightedPICDepositedMoments second =
        weighted_pic_detail::deposit_selected(
            markers,
            mesh,
            [&](const WeightedPICMarker& marker) {
                return batch_of(marker) == 1;
            },
            2.0,
            shape);

    WeightedPICNoiseDiagnostics result;
    const std::size_t cells = mesh.shell_volume.size();
    result.effective_marker_number = full.effective_marker_number;
    result.density_uncertainty.resize(cells, 0.0);
    result.radial_current_uncertainty.resize(cells, 0.0);
    result.kinetic_energy_density_uncertainty.resize(cells, 0.0);
    result.radial_energy_flux_uncertainty.resize(cells, 0.0);
    for (std::size_t i = 0; i < cells; ++i) {
        result.density_uncertainty[i] =
            0.5 * std::abs(first.density[i] - second.density[i]);
        result.radial_current_uncertainty[i] = 0.5 * std::abs(
            first.radial_momentum_density[i]
            - second.radial_momentum_density[i]);
        const double first_energy = 0.5
            * (first.radial_second_moment_density[i]
               + first.tangential_second_moment_density[i]);
        const double second_energy = 0.5
            * (second.radial_second_moment_density[i]
               + second.tangential_second_moment_density[i]);
        result.kinetic_energy_density_uncertainty[i] =
            0.5 * std::abs(first_energy - second_energy);
        result.radial_energy_flux_uncertainty[i] = 0.5 * std::abs(
            first.radial_energy_flux_density[i]
            - second.radial_energy_flux_density[i]);
    }
    result.batch_a_markers = batch_count[0];
    result.batch_b_markers = batch_count[1];
    result.batch_a_mass = static_cast<double>(batch_mass[0]);
    result.batch_b_mass = static_cast<double>(batch_mass[1]);
    return result;
}

/// Converts shell-extensive PIC moments to the line-density quadrature used by
/// VENUS's conservative Poisson solver.  The public PIC density above remains
/// the exact shell-volume average.
// Convert extensive shell sums to the line-density measure expected by the
// Poisson discretization. The public shell-volume density remains separately
// stored, so either representation integrates to the same shell mass.
inline Moments weighted_pic_as_venus_moments(
    const WeightedPICDepositedMoments& deposited,
    const WeightedPICRadialMesh& mesh) {
    const std::size_t cells = mesh.venus_center.size();
    if (deposited.shell_mass.size() != cells
        || deposited.shell_radial_momentum.size() != cells
        || deposited.shell_radial_second_moment.size() != cells
        || deposited.shell_tangential_second_moment.size() != cells
        || mesh.venus_width.size() != cells) {
        throw std::runtime_error("Weighted PIC/VENUS moment bridge size mismatch.");
    }
    Moments moments;
    moments.rho.resize(cells, 0.0);
    moments.p.resize(cells, 0.0);
    moments.pt.resize(cells, 0.0);
    moments.j.resize(cells, 0.0);
    for (std::size_t i = 0; i < cells; ++i) {
        const double measure = 4.0 * PI * mesh.venus_center[i]
            * mesh.venus_center[i] * mesh.venus_width[i];
        if (!(measure > 0.0)) {
            throw std::runtime_error("Weighted PIC/VENUS moment bridge has zero measure.");
        }
        moments.rho[i] = deposited.shell_mass[i] / measure;
        moments.p[i] = deposited.shell_radial_second_moment[i] / measure;
        moments.pt[i] = deposited.shell_tangential_second_moment[i] / measure;
        moments.j[i] = deposited.shell_radial_momentum[i] / measure;
    }
    return moments;
}

inline VlasovPoissonField solve_weighted_pic_poisson_field(
    const WeightedPICDepositedMoments& deposited,
    const WeightedPICRadialMesh& mesh,
    double time,
    const Params& p) {
    if (p.physics_model != "vlasov_poisson"
        || !using_conservative_phase_space_cells(p)
        || mesh.venus_center.size() != static_cast<std::size_t>(p.Nr)) {
        throw std::runtime_error(
            "Weighted PIC field coupling requires Vlasov-Poisson with "
            "the conservative cell geometry enabled.");
    }
    const Moments moments = weighted_pic_as_venus_moments(deposited, mesh);
    return solve_vlasov_poisson_field(
        moments, mesh.venus_center, time, p, deposited.hidden_inner_mass);
}

/// Exact integral of 1/a(t)^2 over one static or power-law expanding VP step.
/// In comoving canonical variables x=r/a and p=a*u, this is the drift factor
/// dx = p * integral(dt/a^2).
// In comoving canonical coordinates, free drift uses integral(dt/a^2).
// The power-law antiderivative uses expm1 near small intervals and its logarithmic
// limit at expansion index alpha=1/2.
inline double weighted_pic_inverse_scale_factor_squared_integral(
    double start_time,
    double end_time,
    const Params& p) {
    if (!(end_time >= start_time) || !std::isfinite(start_time)
        || !std::isfinite(end_time)) {
        throw std::runtime_error(
            "Weighted PIC scale-factor integral received invalid times.");
    }
    const double dt = end_time - start_time;
    if (!using_expanding_vp(p) || dt == 0.0) {
        return dt;
    }
    const VPBackground start = vp_background(start_time, p);
    const VPBackground end = vp_background(end_time, p);
    if (!(start.tau > 0.0) || !(end.tau >= start.tau)
        || !(start.a > 0.0) || !(end.a > 0.0)) {
        throw std::runtime_error(
            "Weighted PIC expansion requires positive background time and scale factor.");
    }
    const double logarithmic_ratio = std::log(end.tau / start.tau);
    const double exponent = 1.0 - 2.0 * p.vp_alpha;
    const double prefactor = start.tau / (start.a * start.a);
    if (std::abs(exponent) < 64.0 * std::numeric_limits<double>::epsilon()) {
        return prefactor * logarithmic_ratio;
    }
    return prefactor * std::expm1(exponent * logarithmic_ratio) / exponent;
}

namespace weighted_pic_detail {

inline void validate_self_consistent_kdk_inputs(
    double time,
    double dt,
    const std::vector<double>& radial_coordinate,
    const std::vector<double>& radius,
    const Params& p,
    unsigned int requested_threads) {
    if (p.physics_model != "vlasov_poisson"
        || !using_kick_drift_kick_integrator(p)
        || using_delayed_inner_return(p)
        || !using_reflecting_outer(p)
        || !(dt >= 0.0) || !std::isfinite(time) || !std::isfinite(dt)
        || requested_threads == 0
        || radius.size() != static_cast<std::size_t>(p.Nr)
        || radial_coordinate.size() != radius.size()) {
        throw std::runtime_error(
            "Self-consistent weighted PIC KDK requires finite times, an "
            "origin-parity or reflecting inner boundary, a reflecting outer "
            "wall, and a matching Vlasov-Poisson KDK grid.");
    }
}

}  // namespace weighted_pic_detail

/// First half of a self-consistent PIC KDK step.  Gravity is kicked in the
/// canonical radial momentum p=a*u using the old Poisson field.  The drift is
/// the exact free central-motion map in comoving coordinates, including the
/// conserved angular momentum, with s=integral(dt/a^2):
///
///   x_new^2 = (x+s*p)^2 + (s*L/x)^2.
///
/// Optional diagnostics for one attempted drift. The caller adds these to a
/// run's budget only after accepting that orbit step. Physical reflecting
/// impacts exclude regular passages through the origin.
struct WeightedPICOrbitDiagnostics {
    std::uint64_t reflecting_wall_impacts = 0;
    double maximum_endpoint_radius = 0.0;
};

/// Integrating the centrifugal motion in this exact drift avoids a timestep
/// singularity at pericentre. Markers end with w storing the endpoint
/// peculiar half-step velocity p_new/a(t+dt).
// Kick canonical momentum with the old field, then drift the complete free
// central orbit analytically in comoving coordinates. The endpoint stores the
// peculiar half-step velocity for a later kick with the newly deposited field.
inline void begin_weighted_pic_self_consistent_kdk_parallel(
    std::vector<WeightedPICMarker>& markers,
    double time,
    double dt,
    const std::vector<double>& radial_coordinate,
    const std::vector<double>& radius,
    const VlasovPoissonField& old_field,
    const Params& p,
    unsigned int requested_threads = std::thread::hardware_concurrency(),
    WeightedPICOrbitDiagnostics* diagnostics = nullptr) {
    weighted_pic_detail::validate_self_consistent_kdk_inputs(
        time, dt, radial_coordinate, radius, p, requested_threads);
    if (markers.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Weighted PIC marker count exceeds the KDK loop limit.");
    }
    const VPBackground start = vp_background(time, p);
    const VPBackground end = vp_background(time + dt, p);
    const double drift = weighted_pic_inverse_scale_factor_squared_integral(
        time, time + dt, p);
    const int workers = static_cast<int>(std::min<std::size_t>(
        requested_threads, std::max<std::size_t>(1, markers.size())));
    std::vector<WeightedPICOrbitDiagnostics> worker_diagnostics(
        diagnostics ? static_cast<std::size_t>(workers) : 0);
    parallel_for_chunks(
        0,
        static_cast<int>(markers.size()),
        workers,
        [&](int first, int last, int worker) {
            WeightedPICOrbitDiagnostics local_diagnostics;
            for (int q = first; q < last; ++q) {
                WeightedPICMarker& marker = markers[static_cast<std::size_t>(q)];
                if (marker.state != WeightedPICMarkerState::active) {
                    throw std::runtime_error(
                        "Self-consistent weighted PIC KDK supports active markers only.");
                }
                const bool radial_origin = using_origin_parity_inner(p)
                    && marker.eta == 0.0;
                if (!(marker.r >= 0.0)
                    || (marker.r == 0.0 && (!radial_origin || p.central_mass != 0.0))) {
                    throw std::runtime_error(
                        "Self-consistent weighted PIC KDK requires L=0 and zero "
                        "central point mass at the origin.");
                }
                const VPSample sample = sample_vp_field(
                    marker.r, radial_coordinate, radius, old_field, p);
                const double x = marker.r / start.a;
                const double angular_momentum = std::sqrt(2.0 * marker.eta);
                const double canonical_force = -start.a * sample.phi_r;
                double canonical_momentum = start.a * marker.w
                    + 0.5 * dt * canonical_force;
                const double radial_axis = x + drift * canonical_momentum;
                const double tangential_axis = radial_origin ? 0.0
                    : drift * angular_momentum / x;
                const double new_x = std::hypot(radial_axis, tangential_axis);
                if (!std::isfinite(new_x)
                    || (!(new_x > 0.0) && using_origin_parity_inner(p) && !radial_origin)
                    || (radial_origin && p.central_mass != 0.0 && radial_axis <= 0.0)) {
                    throw std::runtime_error(
                        "Self-consistent weighted PIC angular drift reached "
                        "a singular central state.");
                }
                if (radial_origin) {
                    // Continue the signed radial trajectory through r=0. An
                    // exact endpoint uses the outgoing radial velocity.
                    if (radial_axis < 0.0) canonical_momentum = -canonical_momentum;
                    else if (radial_axis == 0.0 && dt > 0.0)
                        canonical_momentum = std::fabs(canonical_momentum);
                } else {
                    canonical_momentum = new_x > 0.0 ? (
                        canonical_momentum * radial_axis
                        + angular_momentum / x * tangential_axis) / new_x : 0.0;
                }
                double new_radius = end.a * new_x;
                double new_velocity = canonical_momentum / end.a;
                int wall_impacts = 0;
                // Locate physical wall impacts within the free angular drift.
                // Endpoint mirroring changes L^2/(2r^2) when L is nonzero
                // and can miss an inward crossing followed by an outward one.
                const double initial_momentum = start.a * marker.w
                    + 0.5 * dt * canonical_force;
                const double tangential_momentum = radial_origin ? 0.0 : angular_momentum / x;
                const double speed2 = initial_momentum * initial_momentum
                    + tangential_momentum * tangential_momentum;
                const double nearest_drift = speed2 > 0.0
                    ? std::max(0.0, std::min(drift, -x * initial_momentum / speed2))
                    : 0.0;
                const double minimum_x = std::hypot(
                    x + nearest_drift * initial_momentum,
                    nearest_drift * tangential_momentum);
                const double outer = vp_radial_outer_boundary(radius, p);
                if ((!using_origin_parity_inner(p)
                        && start.a * minimum_x <= p.r_min)
                    || end.a * std::max(x, new_x) >= outer) {
                    new_radius = marker.r;
                    new_velocity = initial_momentum / start.a;
                    if (!advance_central_drift_with_walls(
                            new_radius, new_velocity, angular_momentum, dt,
                            using_expanding_vp(p) ? start.tau : 1.0 + dt,
                            using_expanding_vp(p) ? p.vp_alpha : 0.0,
                            p.r_min, !using_origin_parity_inner(p), outer, true,
                            diagnostics ? &wall_impacts : nullptr)) {
                        throw std::runtime_error(
                            "Self-consistent weighted PIC wall event could not be resolved.");
                    }
                }
                if (diagnostics) {
                    local_diagnostics.reflecting_wall_impacts += wall_impacts;
                    local_diagnostics.maximum_endpoint_radius = std::max(
                        local_diagnostics.maximum_endpoint_radius,
                        std::max(marker.r, new_radius));
                }
                marker.r = new_radius;
                marker.w = new_velocity;
                if (!std::isfinite(marker.r) || !std::isfinite(marker.w)
                    || marker.r < (using_origin_parity_inner(p) ? 0.0 : p.r_min)
                    || marker.r > vp_radial_outer_boundary(radius, p)) {
                    throw std::runtime_error(
                        "Self-consistent weighted PIC KDK drift produced an invalid marker.");
                }
            }
            if (diagnostics) worker_diagnostics[static_cast<std::size_t>(worker)]
                = local_diagnostics;
        });
    if (diagnostics) {
        *diagnostics = {};
        for (const auto& local : worker_diagnostics) {
            diagnostics->reflecting_wall_impacts += local.reflecting_wall_impacts;
            diagnostics->maximum_endpoint_radius = std::max(
                diagnostics->maximum_endpoint_radius, local.maximum_endpoint_radius);
        }
    }
}

/// Completes the endpoint kick of a self-consistent PIC KDK step using the
/// Poisson field reconstructed after the drift/deposition.  Marker positions,
/// masses, and angular momenta are unchanged.
// The endpoint field supplies the second half kick at the endpoint scale
// factor. Particle positions, weights and eta retain their drifted values.
inline void finish_weighted_pic_self_consistent_kdk_parallel(
    std::vector<WeightedPICMarker>& markers,
    double endpoint_time,
    double dt,
    const std::vector<double>& radial_coordinate,
    const std::vector<double>& radius,
    const VlasovPoissonField& new_field,
    const Params& p,
    unsigned int requested_threads = std::thread::hardware_concurrency()) {
    weighted_pic_detail::validate_self_consistent_kdk_inputs(
        endpoint_time - dt,
        dt,
        radial_coordinate,
        radius,
        p,
        requested_threads);
    if (markers.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Weighted PIC marker count exceeds the KDK loop limit.");
    }
    const VPBackground end = vp_background(endpoint_time, p);
    const int workers = static_cast<int>(std::min<std::size_t>(
        requested_threads, std::max<std::size_t>(1, markers.size())));
    parallel_for_chunks(
        0,
        static_cast<int>(markers.size()),
        workers,
        [&](int first, int last, int) {
            for (int q = first; q < last; ++q) {
                WeightedPICMarker& marker = markers[static_cast<std::size_t>(q)];
                if (marker.state != WeightedPICMarkerState::active) {
                    throw std::runtime_error(
                        "Self-consistent weighted PIC KDK supports active markers only.");
                }
                const VPSample sample = sample_vp_field(
                    marker.r, radial_coordinate, radius, new_field, p);
                const double canonical_force = -end.a * sample.phi_r;
                const double canonical_momentum = end.a * marker.w
                    + 0.5 * dt * canonical_force;
                marker.w = canonical_momentum / end.a;
                if (!std::isfinite(marker.w)) {
                    throw std::runtime_error(
                        "Self-consistent weighted PIC KDK kick produced an invalid marker.");
                }
            }
        });
}

enum class WeightedPICOrbitDisposition {
    active,
    entered_delayed_inner,
    escaped,
};

struct WeightedPICOrbitOutcome {
    WeightedPICOrbitDisposition disposition =
        WeightedPICOrbitDisposition::active;
    double r = 0.0;
    double w = 0.0;
    double elapsed_time = 0.0;
    double inner_boundary_speed = 0.0;
    double inner_round_trip_time = 0.0;
};

namespace weighted_pic_detail {

template <class OrbitPusher>
// Consume the requested physical interval across active motion and any
// unresolved inner return. A delayed particle retains its mass/eta while its
// return clock counts down; an escaped particle stays in the escape ledger.
inline void advance_one_weighted_pic_marker(
    WeightedPICMarker& marker,
    double dt,
    OrbitPusher& pusher,
    int maximum_events_per_marker) {
    validate_weighted_pic_marker(marker, "Weighted PIC advance input");
    if (marker.state == WeightedPICMarkerState::escaped || dt == 0.0) {
        return;
    }
    const double invariant_mass = marker.mass;
    const double invariant_eta = marker.eta;
    double remaining = dt;
    int events = 0;
    while (remaining > 0.0) {
        if (++events > maximum_events_per_marker) {
            throw std::runtime_error(
                "Weighted PIC orbit exceeded the per-step boundary-event limit.");
        }
        if (marker.state == WeightedPICMarkerState::delayed_inner) {
            if (marker.remaining_inner_time > remaining) {
                marker.remaining_inner_time -= remaining;
                remaining = 0.0;
                break;
            }
            remaining -= marker.remaining_inner_time;
            marker.remaining_inner_time = 0.0;
            marker.state = WeightedPICMarkerState::active;
            marker.w = marker.inner_boundary_speed;
            if (remaining == 0.0) {
                break;
            }
        }

        const WeightedPICOrbitOutcome outcome = pusher(marker, remaining);
        if (!std::isfinite(outcome.r) || !std::isfinite(outcome.w)
            || !(outcome.elapsed_time >= 0.0)
            || outcome.elapsed_time > remaining
                + 64.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, remaining)) {
            throw std::runtime_error("Weighted PIC pusher returned an invalid outcome.");
        }
        marker.r = outcome.r;
        marker.w = outcome.w;
        if (outcome.disposition == WeightedPICOrbitDisposition::active) {
            const double completion_tolerance =
                64.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, remaining);
            if (std::abs(outcome.elapsed_time - remaining)
                > completion_tolerance) {
                throw std::runtime_error(
                    "Weighted PIC active pusher outcome did not consume its timestep.");
            }
            remaining = 0.0;
            break;
        }
        if (outcome.disposition == WeightedPICOrbitDisposition::escaped) {
            marker.state = WeightedPICMarkerState::escaped;
            remaining = 0.0;
            break;
        }
        if (!(outcome.inner_boundary_speed >= 0.0)
            || !std::isfinite(outcome.inner_boundary_speed)
            || !(outcome.inner_round_trip_time >= 0.0)
            || !std::isfinite(outcome.inner_round_trip_time)) {
            throw std::runtime_error(
                "Weighted PIC pusher returned invalid delayed-inner data.");
        }
        const double after_crossing = std::max(
            0.0, remaining - outcome.elapsed_time);
        marker.inner_boundary_speed = outcome.inner_boundary_speed;
        if (outcome.inner_round_trip_time > after_crossing) {
            marker.state = WeightedPICMarkerState::delayed_inner;
            marker.remaining_inner_time =
                outcome.inner_round_trip_time - after_crossing;
            remaining = 0.0;
        } else {
            marker.state = WeightedPICMarkerState::active;
            marker.w = marker.inner_boundary_speed;
            marker.remaining_inner_time = 0.0;
            remaining = after_crossing - outcome.inner_round_trip_time;
        }
    }
    if (marker.mass != invariant_mass || marker.eta != invariant_eta) {
        throw std::runtime_error(
            "Weighted PIC pusher changed collisionless marker mass or eta.");
    }
}

}  // namespace weighted_pic_detail

/// Advances markers through a pusher abstraction that may report an inner
/// crossing or escape.  Delayed carriers retain their marker mass and eta.
template <class OrbitPusher>
inline void advance_weighted_pic_markers(
    std::vector<WeightedPICMarker>& markers,
    double dt,
    OrbitPusher&& pusher,
    int maximum_events_per_marker = 16) {
    if (!(dt >= 0.0) || !std::isfinite(dt)
        || maximum_events_per_marker < 1) {
        throw std::runtime_error("Weighted PIC advance received invalid controls.");
    }
    for (WeightedPICMarker& marker : markers) {
        weighted_pic_detail::advance_one_weighted_pic_marker(
            marker, dt, pusher, maximum_events_per_marker);
    }
}

/// Deterministic threaded marker advance.  The pusher must be safe for
/// concurrent read-only calls.  Contiguous ownership ensures each marker is
/// written by exactly one worker; deposition order and marker order do not
/// change.
template <class OrbitPusher>
inline void advance_weighted_pic_markers_parallel(
    std::vector<WeightedPICMarker>& markers,
    double dt,
    OrbitPusher&& pusher,
    unsigned int requested_threads = std::thread::hardware_concurrency(),
    int maximum_events_per_marker = 16) {
    if (!(dt >= 0.0) || !std::isfinite(dt)
        || maximum_events_per_marker < 1 || requested_threads == 0) {
        throw std::runtime_error(
            "Weighted PIC parallel advance received invalid controls.");
    }
    const int worker_count = static_cast<int>(std::min<std::size_t>({
        requested_threads,
        std::max<std::size_t>(1, markers.size()),
        static_cast<std::size_t>(std::numeric_limits<int>::max()),
    }));
    if (worker_count == 1 || markers.size() < 2) {
        advance_weighted_pic_markers(
            markers, dt, pusher, maximum_events_per_marker);
        return;
    }

    // A small fraction of pericentre trajectories invoke the much more
    // expensive Levi--Civita retry.  Dynamic marker blocks keep those orbits
    // distributed across the physical cores; each marker remains an
    // independent deterministic calculation, so scheduling cannot change
    // the evolved state.
    constexpr int orbit_chunk_markers = 256;
    parallel_for_dynamic_chunks(
        0,
        static_cast<int>(markers.size()),
        worker_count,
        orbit_chunk_markers,
        [&](int begin, int end, int) {
            for (int marker = begin; marker < end; ++marker) {
                weighted_pic_detail::advance_one_weighted_pic_marker(
                    markers[static_cast<std::size_t>(marker)],
                    dt,
                    pusher,
                    maximum_events_per_marker);
            }
        });
}

inline WeightedPICOrbitOutcome weighted_pic_delayed_inner_outcome(
    const WeightedPICMarker& marker_at_face,
    double elapsed_time,
    double hidden_mass,
    const Params& p) {
    const double speed = std::abs(marker_at_face.w);
    const double angular_momentum = std::sqrt(2.0 * marker_at_face.eta);
    return {
        WeightedPICOrbitDisposition::entered_delayed_inner,
        marker_at_face.r,
        marker_at_face.w,
        elapsed_time,
        speed,
        delayed_inner_uniform_cell_round_trip_time(
            speed, angular_momentum, hidden_mass, p),
    };
}

/// Adapter for point-characteristic transport. Its caller supplies the full
/// interval; delayed return instead uses advance_weighted_pic_markers with a
/// pusher that locates the first inner-face crossing and its elapsed time.
inline void advance_weighted_pic_markers_with_venus(
    std::vector<WeightedPICMarker>& markers,
    double dt,
    const std::vector<double>& radial_coordinate,
    const std::vector<double>& radius,
    const VlasovPoissonField& field,
    const Params& p,
    const CharacteristicIntegratorPlan& plan) {
    if (p.physics_model != "vlasov_poisson") {
        throw std::runtime_error(
            "Weighted PIC VENUS orbit adapter requires Vlasov-Poisson.");
    }
    if (using_delayed_inner_return(p)) {
        throw std::runtime_error(
            "Weighted PIC delayed_return needs a public first-crossing event "
            "from the VENUS characteristic integrator; use the generic event pusher.");
    }
    advance_weighted_pic_markers(
        markers,
        dt,
        [&](const WeightedPICMarker& marker, double local_dt) {
            const CharacteristicState state = advance_vp_characteristic(
                marker.r,
                marker.w,
                std::sqrt(2.0 * marker.eta),
                radial_coordinate,
                radius,
                field,
                local_dt,
                p,
                plan,
                true,
                false);
            const bool escaped = !using_reflecting_outer(p)
                && state.r > vp_radial_outer_boundary(radius, p);
            return WeightedPICOrbitOutcome{
                escaped ? WeightedPICOrbitDisposition::escaped
                        : WeightedPICOrbitDisposition::active,
                state.r,
                state.w,
                local_dt,
                0.0,
                0.0,
            };
        });
}

inline void advance_weighted_pic_markers_with_venus_parallel(
    std::vector<WeightedPICMarker>& markers,
    double dt,
    const std::vector<double>& radial_coordinate,
    const std::vector<double>& radius,
    const VlasovPoissonField& field,
    const Params& p,
    const CharacteristicIntegratorPlan& plan,
    unsigned int requested_threads = std::thread::hardware_concurrency()) {
    if (p.physics_model != "vlasov_poisson") {
        throw std::runtime_error(
            "Weighted PIC parallel VENUS adapter requires Vlasov-Poisson.");
    }
    if (using_delayed_inner_return(p)) {
        throw std::runtime_error(
            "Weighted PIC delayed_return needs a public first-crossing event "
            "from the VENUS characteristic integrator; use the generic event pusher.");
    }
    advance_weighted_pic_markers_parallel(
        markers,
        dt,
        [&](const WeightedPICMarker& marker, double local_dt) {
            const CharacteristicState state = advance_vp_characteristic(
                marker.r,
                marker.w,
                std::sqrt(2.0 * marker.eta),
                radial_coordinate,
                radius,
                field,
                local_dt,
                p,
                plan,
                true,
                false);
            const bool escaped = !using_reflecting_outer(p)
                && state.r > vp_radial_outer_boundary(radius, p);
            return WeightedPICOrbitOutcome{
                escaped ? WeightedPICOrbitDisposition::escaped
                        : WeightedPICOrbitDisposition::active,
                state.r,
                state.w,
                local_dt,
                0.0,
                0.0,
            };
        },
        requested_threads);
}

struct WeightedPICRestartState {
    std::vector<WeightedPICMarker> markers;
    WeightedPICSamplerState sampler;
    WeightedPICCollisionSamplerState collision_sampler;
    double potential_event_age = 0.0;
    double collision_event_age = 0.0;
    // Local physical clocks used by the adaptive collision-application
    // scheduler.  Infinity means that no finite estimate is available yet.
    double collision_timescale_estimate =
        std::numeric_limits<double>::infinity();
    double potential_timescale_estimate =
        std::numeric_limits<double>::infinity();
    double time = 0.0;
    std::int64_t step = 0;
    // Static Boltzmann event evolution holds the gravitational field fixed
    // between potential events. Version 7 preserves that field and the
    // numerical/physical controls needed to continue its event schedule.
    // Other WPIC drivers continue to write their existing version-6 state.
    bool has_frozen_field = false;
    VlasovPoissonField frozen_field;
    std::string frozen_field_contract;
};

namespace weighted_pic_detail {

template <class T>
inline void write_scalar(std::ofstream& output, const T& value) {
    static_assert(std::is_trivially_copyable<T>::value, "PIC restart scalar");
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!output) {
        throw std::runtime_error("Failed while writing weighted PIC restart.");
    }
}

template <class T>
inline T read_scalar(std::ifstream& input) {
    static_assert(std::is_trivially_copyable<T>::value, "PIC restart scalar");
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("Unexpected end of weighted PIC restart.");
    }
    return value;
}

inline void validate_unique_marker_ids(
    const std::vector<WeightedPICMarker>& markers,
    const char* context) {
    std::vector<std::uint64_t> ids;
    ids.reserve(markers.size());
    for (const WeightedPICMarker& marker : markers) {
        validate_weighted_pic_marker(marker, context);
        ids.push_back(marker.id);
    }
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
        throw std::runtime_error(
            std::string(context) + " contains duplicate marker IDs.");
    }
}

// A held field is part of a two-clock restart. Validate the complete radial
// source/potential arrays, ordered faces and finite background metadata before
// accepting this versioned state.
inline void validate_restart_frozen_field(const VlasovPoissonField& field) {
    const std::size_t n = field.phi.size();
    if (n == 0 || n > 10000000 || field.radial_edge.size() != n + 1
        || field.source_mass_edge.size() != n + 1
        || field.phi_edge.size() != n + 1
        || field.source_mass.size() != n || field.phi_r.size() != n
        || field.source_line_density.size() != n
        || field.source_line_density_linear.size() != n
        || field.source_line_density_quadratic.size() != n) {
        throw std::runtime_error("Invalid frozen Poisson field dimensions in restart.");
    }
    for (const auto* values : {&field.source_mass, &field.phi, &field.phi_r,
            &field.radial_edge, &field.source_mass_edge,
            &field.source_line_density, &field.source_line_density_linear,
            &field.source_line_density_quadratic, &field.phi_edge}) {
        for (double value : *values) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("Nonfinite frozen Poisson field in restart.");
            }
        }
    }
    for (std::size_t i = 1; i < field.radial_edge.size(); ++i) {
        if (!(field.radial_edge[i] > field.radial_edge[i - 1])) {
            throw std::runtime_error("Nonincreasing frozen Poisson mesh in restart.");
        }
    }
    for (double value : {field.background.tau, field.background.a,
            field.background.H, field.background.rho_bar, field.matter_mass,
            field.outer_source_mass, field.hidden_carrier_mass,
            field.fixed_central_mass, field.min_gap}) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Nonfinite frozen Poisson metadata in restart.");
        }
    }
    if (field.radial_edge.front() < 0.0 || !(field.background.tau > 0.0)
        || !(field.background.a > 0.0)) {
        throw std::runtime_error("Invalid frozen Poisson geometry in restart.");
    }
}

inline void write_restart_frozen_field(
    std::ofstream& output, const WeightedPICRestartState& state) {
    const auto& field = state.frozen_field;
    write_scalar(output, static_cast<std::uint64_t>(state.frozen_field_contract.size()));
    output.write(state.frozen_field_contract.data(), state.frozen_field_contract.size());
    for (const auto* values : {&field.source_mass, &field.phi, &field.phi_r,
            &field.radial_edge, &field.source_mass_edge,
            &field.source_line_density, &field.source_line_density_linear,
            &field.source_line_density_quadratic, &field.phi_edge}) {
        write_scalar(output, static_cast<std::uint64_t>(values->size()));
        output.write(reinterpret_cast<const char*>(values->data()),
                     values->size() * sizeof(double));
    }
    for (double value : {field.background.tau, field.background.a,
            field.background.H, field.background.rho_bar, field.matter_mass,
            field.outer_source_mass, field.hidden_carrier_mass,
            field.fixed_central_mass, field.min_gap}) {
        write_scalar(output, value);
    }
}

inline void read_restart_frozen_field(
    std::ifstream& input, WeightedPICRestartState& state) {
    auto& field = state.frozen_field;
    const auto length = read_scalar<std::uint64_t>(input);
    if (length == 0 || length > 65536) {
        throw std::runtime_error("Invalid frozen-field restart contract length.");
    }
    state.frozen_field_contract.resize(static_cast<std::size_t>(length));
    input.read(&state.frozen_field_contract[0], length);
    for (auto* values : {&field.source_mass, &field.phi, &field.phi_r,
            &field.radial_edge, &field.source_mass_edge,
            &field.source_line_density, &field.source_line_density_linear,
            &field.source_line_density_quadratic, &field.phi_edge}) {
        const auto count = read_scalar<std::uint64_t>(input);
        if (count == 0 || count > 10000001) {
            throw std::runtime_error("Invalid frozen-field restart vector length.");
        }
        values->resize(static_cast<std::size_t>(count));
        input.read(reinterpret_cast<char*>(values->data()), count * sizeof(double));
        if (!input) throw std::runtime_error("Incomplete frozen Poisson field in restart.");
    }
    for (double* value : {&field.background.tau, &field.background.a,
            &field.background.H, &field.background.rho_bar, &field.matter_mass,
            &field.outer_source_mass, &field.hidden_carrier_mass,
            &field.fixed_central_mass, &field.min_gap}) {
        *value = read_scalar<double>(input);
    }
    validate_restart_frozen_field(field);
    state.has_frozen_field = true;
}

}  // namespace weighted_pic_detail

/// Writes a PIC-only, explicit-field checkpoint through an atomic POSIX rename.
// Write time, independent random streams, event ages and complete particle
// records in a padding-free field order. Version 7 additionally stores the held
// field and its controls, preserving the potential-event schedule.
inline void write_weighted_pic_restart(
    const std::filesystem::path& path,
    const WeightedPICRestartState& state,
    bool validate_marker_ids = true) {
    if (!(state.time >= 0.0) || !std::isfinite(state.time) || state.step < 0
        || !(state.potential_event_age >= 0.0)
        || !std::isfinite(state.potential_event_age)
        || !(state.collision_event_age >= 0.0)
        || !std::isfinite(state.collision_event_age)
        || !(state.collision_timescale_estimate > 0.0)
        || std::isnan(state.collision_timescale_estimate)
        || !(state.potential_timescale_estimate > 0.0)
        || std::isnan(state.potential_timescale_estimate)) {
        throw std::runtime_error("Weighted PIC restart has invalid time metadata.");
    }
    if (validate_marker_ids) {
        weighted_pic_detail::validate_unique_marker_ids(
            state.markers, "Weighted PIC restart write");
    }
    if (state.has_frozen_field) {
        weighted_pic_detail::validate_restart_frozen_field(state.frozen_field);
        if (state.frozen_field_contract.empty()
            || state.frozen_field_contract.size() > 65536) {
            throw std::runtime_error("Invalid frozen-field restart contract.");
        }
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    const std::filesystem::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Could not open weighted PIC restart for writing: " + temporary.string());
    }
    const char magic[16] = {
        'V', 'E', 'N', 'U', 'S', 'W', 'V', 'P', 'I', 'C', '1', '\0'};
    output.write(magic, sizeof(magic));
    weighted_pic_detail::write_scalar(
        output, std::uint32_t{state.has_frozen_field ? 7u : 6u});
    weighted_pic_detail::write_scalar(output, UINT32_C(0x01020304));
    weighted_pic_detail::write_scalar(
        output, static_cast<std::uint64_t>(state.markers.size()));
    weighted_pic_detail::write_scalar(output, state.time);
    weighted_pic_detail::write_scalar(output, state.step);
    weighted_pic_detail::write_scalar(output, state.sampler.seed);
    weighted_pic_detail::write_scalar(output, state.sampler.next_counter);
    weighted_pic_detail::write_scalar(output, state.collision_sampler.seed);
    weighted_pic_detail::write_scalar(
        output, state.collision_sampler.next_counter);
    weighted_pic_detail::write_scalar(output, state.potential_event_age);
    weighted_pic_detail::write_scalar(output, state.collision_event_age);
    weighted_pic_detail::write_scalar(
        output, state.collision_timescale_estimate);
    weighted_pic_detail::write_scalar(
        output, state.potential_timescale_estimate);
    // Version 6 stores an 81-byte, padding-free record.  Packing a block in
    // memory preserves the exact field order while avoiding eleven small
    // ostream calls per marker, which otherwise dominates large checkpoints.
    constexpr std::size_t marker_record_bytes =
        8 * sizeof(double) + 2 * sizeof(std::uint64_t)
        + sizeof(std::uint8_t);
    constexpr std::size_t markers_per_block = 16384;
    std::vector<char> marker_buffer(
        marker_record_bytes * markers_per_block);
    for (std::size_t begin = 0; begin < state.markers.size();
         begin += markers_per_block) {
        const std::size_t end = std::min(
            state.markers.size(), begin + markers_per_block);
        char* cursor = marker_buffer.data();
        auto pack = [&](const auto& value) {
            std::memcpy(cursor, &value, sizeof(value));
            cursor += sizeof(value);
        };
        for (std::size_t i = begin; i < end; ++i) {
            const WeightedPICMarker& marker = state.markers[i];
            pack(marker.mass);
            pack(marker.r);
            pack(marker.w);
            pack(marker.eta);
            pack(marker.internal_energy);
            pack(marker.internal_velocity_dispersion);
            pack(marker.id);
            pack(marker.source_cell);
            const std::uint8_t marker_state =
                static_cast<std::uint8_t>(marker.state);
            pack(marker_state);
            pack(marker.inner_boundary_speed);
            pack(marker.remaining_inner_time);
        }
        const std::streamsize bytes = static_cast<std::streamsize>(
            cursor - marker_buffer.data());
        output.write(marker_buffer.data(), bytes);
        if (!output) {
            throw std::runtime_error(
                "Failed while writing weighted PIC restart marker block.");
        }
    }
    if (state.has_frozen_field) {
        weighted_pic_detail::write_restart_frozen_field(output, state);
    }
    output.flush();
    if (!output) {
        throw std::runtime_error(
            "Failed to flush weighted PIC restart: " + temporary.string());
    }
    output.close();
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        throw std::runtime_error(
            "Could not install weighted PIC restart: " + path.string());
    }
}

// Read only declared format versions, reconstruct fields introduced by each
// version and validate unique IDs and final EOF. The byte-order sentinel identifies
// the host binary scalar representation used by the checkpoint.
inline WeightedPICRestartState read_weighted_pic_restart(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "Could not open weighted PIC restart: " + path.string());
    }
    char magic[16] = {};
    input.read(magic, sizeof(magic));
    const char expected_magic[16] = {
        'V', 'E', 'N', 'U', 'S', 'W', 'V', 'P', 'I', 'C', '1', '\0'};
    if (!input || std::memcmp(magic, expected_magic, sizeof(magic)) != 0) {
        throw std::runtime_error("Invalid weighted PIC restart magic: " + path.string());
    }
    const std::uint32_t version = weighted_pic_detail::read_scalar<std::uint32_t>(input);
    const std::uint32_t endian = weighted_pic_detail::read_scalar<std::uint32_t>(input);
    if ((version != 1 && version != 2 && version != 3 && version != 4
            && version != 5 && version != 6 && version != 7)
        || endian != UINT32_C(0x01020304)) {
        throw std::runtime_error(
            "Unsupported weighted PIC restart version or byte order.");
    }
    const std::uint64_t marker_count =
        weighted_pic_detail::read_scalar<std::uint64_t>(input);
    if (marker_count > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Weighted PIC restart marker count overflows size_t.");
    }
    WeightedPICRestartState state;
    state.time = weighted_pic_detail::read_scalar<double>(input);
    state.step = weighted_pic_detail::read_scalar<std::int64_t>(input);
    state.sampler.seed = weighted_pic_detail::read_scalar<std::uint64_t>(input);
    state.sampler.next_counter =
        weighted_pic_detail::read_scalar<std::uint64_t>(input);
    if (version >= 2) {
        state.collision_sampler.seed =
            weighted_pic_detail::read_scalar<std::uint64_t>(input);
        state.collision_sampler.next_counter =
            weighted_pic_detail::read_scalar<std::uint64_t>(input);
    }
    if (version >= 3) {
        state.potential_event_age =
            weighted_pic_detail::read_scalar<double>(input);
        state.collision_event_age =
            weighted_pic_detail::read_scalar<double>(input);
    }
    if (version >= 6) {
        state.collision_timescale_estimate =
            weighted_pic_detail::read_scalar<double>(input);
        state.potential_timescale_estimate =
            weighted_pic_detail::read_scalar<double>(input);
    }
    state.markers.resize(static_cast<std::size_t>(marker_count));
    for (WeightedPICMarker& marker : state.markers) {
        marker.mass = weighted_pic_detail::read_scalar<double>(input);
        marker.r = weighted_pic_detail::read_scalar<double>(input);
        marker.w = weighted_pic_detail::read_scalar<double>(input);
        marker.eta = weighted_pic_detail::read_scalar<double>(input);
        if (version >= 4) {
            marker.internal_energy =
                weighted_pic_detail::read_scalar<double>(input);
        }
        if (version >= 5) {
            marker.internal_velocity_dispersion =
                weighted_pic_detail::read_scalar<double>(input);
        } else if (marker.internal_energy > 0.0) {
            marker.internal_velocity_dispersion = std::sqrt(
                2.0 * marker.internal_energy / (3.0 * marker.mass));
        }
        marker.id = weighted_pic_detail::read_scalar<std::uint64_t>(input);
        marker.source_cell =
            weighted_pic_detail::read_scalar<std::uint64_t>(input);
        const std::uint8_t marker_state =
            weighted_pic_detail::read_scalar<std::uint8_t>(input);
        if (marker_state > static_cast<std::uint8_t>(
                WeightedPICMarkerState::escaped)) {
            throw std::runtime_error("Weighted PIC restart has an invalid marker state.");
        }
        marker.state = static_cast<WeightedPICMarkerState>(marker_state);
        marker.inner_boundary_speed =
            weighted_pic_detail::read_scalar<double>(input);
        marker.remaining_inner_time =
            weighted_pic_detail::read_scalar<double>(input);
    }
    if (version == 7) {
        weighted_pic_detail::read_restart_frozen_field(input, state);
    }
    if (!(state.time >= 0.0) || !std::isfinite(state.time) || state.step < 0
        || !(state.potential_event_age >= 0.0)
        || !std::isfinite(state.potential_event_age)
        || !(state.collision_event_age >= 0.0)
        || !std::isfinite(state.collision_event_age)
        || !(state.collision_timescale_estimate > 0.0)
        || std::isnan(state.collision_timescale_estimate)
        || !(state.potential_timescale_estimate > 0.0)
        || std::isnan(state.potential_timescale_estimate)) {
        throw std::runtime_error("Weighted PIC restart has invalid time metadata.");
    }
    weighted_pic_detail::validate_unique_marker_ids(
        state.markers, "Weighted PIC restart read");
    char trailing = 0;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error("Weighted PIC restart has trailing data.");
    }
    if (!input.eof()) {
        throw std::runtime_error("Failed while closing weighted PIC restart read.");
    }
    return state;
}

}  // namespace ev
