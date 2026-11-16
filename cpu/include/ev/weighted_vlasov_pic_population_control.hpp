#pragma once

// Refine a collisionless empirical particle distribution with equal clones.
// When live marker count falls below the declared threshold, each active or
// delayed-return marker becomes k colocated daughters with W/k and absolute
// internal energy/k. Their specific orbit, internal dispersion and boundary
// state are retained; escaped records remain unchanged. The operation checks
// mass, momentum, angular and energy sums before replacing the population.
//
// This is numerical quadrature refinement, distinct from physical fragmentation.
// For the subhalo area A_ij=A*(sqrt(m_i)+sqrt(m_j))^2/4, a common factor k
// preserves off-diagonal continuum n*A_ij. It also creates same-parent pairs,
// whose additional mass-processing term scales as A*m^2*(k-1)/k. Those pairs
// need lineage exclusion or a separately justified conservative resample in a
// collisional model. The expanding collisional driver rejects this option;
// the present helper supports collisionless sampling studies.

#include "ev/weighted_vlasov_pic.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace ev {

struct WeightedPICPopulationControlOptions {
    // Zero disables population control.  Otherwise refinement is applied when
    // the number of active plus delayed-return markers is below this value.
    std::size_t trigger_live_markers = 0;
    // Common daughter multiplicity.  A common factor preserves unequal-mass
    // ratios and the off-diagonal homogeneous pair-kernel scaling, but does
    // not remove the same-parent diagonal bias discussed above.
    int refinement_factor = 2;
};

struct WeightedPICPopulationControlDiagnostics {
    bool applied = false;
    std::size_t markers_before = 0;
    std::size_t markers_after = 0;
    std::size_t live_markers_before = 0;
    std::size_t live_markers_after = 0;
    std::size_t escaped_markers = 0;
    std::size_t injected_markers = 0;
    int refinement_factor = 1;
    double mass_before = 0.0;
    double mass_after = 0.0;
    double radial_momentum_before = 0.0;
    double radial_momentum_after = 0.0;
    double angular_moment_before = 0.0;
    double angular_moment_after = 0.0;
    double kinetic_energy_before = 0.0;
    double kinetic_energy_after = 0.0;
    double internal_energy_before = 0.0;
    double internal_energy_after = 0.0;
    double maximum_relative_invariant_residual = 0.0;
};

namespace weighted_pic_population_control_detail {

struct Invariants {
    long double mass = 0.0L;
    long double radial_momentum = 0.0L;
    long double angular_moment = 0.0L;
    long double kinetic_energy = 0.0L;
    long double internal_energy = 0.0L;
};

inline bool is_live(const WeightedPICMarker& marker) {
    return marker.state != WeightedPICMarkerState::escaped;
}

// Measure actual stored represented weights and momenta, including
// internal energy. Orbital kinetic energy is formed only for active
// markers; delayed/escaped lifecycle bookkeeping retains its own meaning.
inline Invariants invariants(
    const std::vector<WeightedPICMarker>& markers) {
    Invariants result;
    for (const WeightedPICMarker& marker : markers) {
        validate_weighted_pic_marker(
            marker, "Weighted PIC population-control invariant");
        const long double mass = marker.mass;
        result.mass += mass;
        result.radial_momentum += mass * marker.w;
        result.angular_moment += mass * marker.eta;
        if (marker.state == WeightedPICMarkerState::active) {
            const long double radius = std::max(
                static_cast<long double>(marker.r), 1.0e-300L);
            const long double tangential_speed_squared =
                2.0L * marker.eta / (radius * radius);
            result.kinetic_energy += 0.5L * mass
                * (static_cast<long double>(marker.w) * marker.w
                   + tangential_speed_squared);
        }
        result.internal_energy += marker.internal_energy;
    }
    return result;
}

inline double relative_residual(long double before, long double after) {
    const long double scale = std::max({
        1.0e-300L, std::abs(before), std::abs(after)});
    return static_cast<double>(std::abs(after - before) / scale);
}

}  // namespace weighted_pic_population_control_detail

/// Refine the complete live empirical distribution by one common factor.
///
/// No random number is consumed.  Restarting from the refined marker vector
/// therefore needs no additional sampler state.  Original IDs are retained by
/// the first daughter and all additional daughters receive monotonically
/// increasing IDs above the previous maximum.  This helper does not identify
/// same-parent daughters to the collision operator and must not be enabled in
/// a collisional production run.
inline WeightedPICPopulationControlDiagnostics
refine_weighted_pic_population_if_needed(
    std::vector<WeightedPICMarker>& markers,
    const WeightedPICPopulationControlOptions& options) {
    using namespace weighted_pic_population_control_detail;
    if (options.refinement_factor < 2
        || options.refinement_factor > 64) {
        throw std::runtime_error(
            "Weighted PIC population refinement factor must be in [2,64].");
    }

    WeightedPICPopulationControlDiagnostics diagnostics;
    diagnostics.markers_before = markers.size();
    diagnostics.refinement_factor = options.refinement_factor;
    for (const WeightedPICMarker& marker : markers) {
        validate_weighted_pic_marker(
            marker, "Weighted PIC population-control input");
        if (is_live(marker)) {
            ++diagnostics.live_markers_before;
        } else {
            ++diagnostics.escaped_markers;
        }
    }
    diagnostics.markers_after = diagnostics.markers_before;
    diagnostics.live_markers_after = diagnostics.live_markers_before;
    // Zero disables refinement. Otherwise only a live count strictly
    // below the configured threshold activates the same multiplicity for
    // every live packet; a selective heavy-packet split changes mass ratios.
    if (options.trigger_live_markers == 0
        || diagnostics.live_markers_before >= options.trigger_live_markers) {
        return diagnostics;
    }
    if (diagnostics.live_markers_before == 0) {
        throw std::runtime_error(
            "Weighted PIC population control cannot refine an empty live "
            "distribution.");
    }

    const std::size_t factor =
        static_cast<std::size_t>(options.refinement_factor);
    if (diagnostics.live_markers_before
        > (std::numeric_limits<std::size_t>::max()
           - diagnostics.escaped_markers) / factor) {
        throw std::runtime_error(
            "Weighted PIC population-control marker count overflows size_t.");
    }
    // Preflight the population size before allocating daughters. Their
    // count and monotonically increasing identities must both fit their
    // integer storage domains.
    const std::size_t refined_size =
        diagnostics.escaped_markers
        + factor * diagnostics.live_markers_before;
    diagnostics.injected_markers =
        refined_size - diagnostics.markers_before;

    std::uint64_t maximum_id = 0;
    std::unordered_set<std::uint64_t> existing_id;
    existing_id.reserve(markers.size());
    for (const WeightedPICMarker& marker : markers) {
        maximum_id = std::max(maximum_id, marker.id);
        if (!existing_id.insert(marker.id).second) {
            throw std::runtime_error(
                "Weighted PIC population control found duplicate marker IDs.");
        }
    }
    if (diagnostics.injected_markers
        > std::numeric_limits<std::uint64_t>::max() - maximum_id) {
        throw std::runtime_error(
            "Weighted PIC population-control marker IDs overflow uint64.");
    }

    // Measure the source distribution before creating a new vector.
    // The caller retains its entire old population if a later daughter
    // validation or invariant audit fails.
    const Invariants before = invariants(markers);
    std::vector<WeightedPICMarker> refined;
    refined.reserve(refined_size);
    std::uint64_t next_id = maximum_id + 1;
    for (const WeightedPICMarker& marker : markers) {
        if (!is_live(marker)) {
            refined.push_back(marker);
            continue;
        }
        // Retain the first daughter ID and assign fresh IDs to additional
        // daughters. Dividing extensive mass/internal energy preserves their
        // specific internal state and the instantaneous empirical moments.
        for (std::size_t child = 0; child < factor; ++child) {
            WeightedPICMarker daughter = marker;
            daughter.mass /= static_cast<double>(factor);
            daughter.internal_energy /= static_cast<double>(factor);
            if (child != 0) {
                daughter.id = next_id++;
            }
            validate_weighted_pic_marker(
                daughter, "Weighted PIC population-control daughter");
            refined.push_back(daughter);
        }
    }
    const Invariants after = invariants(refined);
    diagnostics.mass_before = static_cast<double>(before.mass);
    diagnostics.mass_after = static_cast<double>(after.mass);
    diagnostics.radial_momentum_before =
        static_cast<double>(before.radial_momentum);
    diagnostics.radial_momentum_after =
        static_cast<double>(after.radial_momentum);
    diagnostics.angular_moment_before =
        static_cast<double>(before.angular_moment);
    diagnostics.angular_moment_after =
        static_cast<double>(after.angular_moment);
    diagnostics.kinetic_energy_before =
        static_cast<double>(before.kinetic_energy);
    diagnostics.kinetic_energy_after =
        static_cast<double>(after.kinetic_energy);
    diagnostics.internal_energy_before =
        static_cast<double>(before.internal_energy);
    diagnostics.internal_energy_after =
        static_cast<double>(after.internal_energy);
    // Audit all stated budgets on the actual stored double daughter
    // weights, with extended-precision sums. This establishes instantaneous
    // quadrature conservation, not equality of future stochastic encounters.
    diagnostics.maximum_relative_invariant_residual = std::max({
        relative_residual(before.mass, after.mass),
        relative_residual(
            before.radial_momentum, after.radial_momentum),
        relative_residual(before.angular_moment, after.angular_moment),
        relative_residual(before.kinetic_energy, after.kinetic_energy),
        relative_residual(before.internal_energy, after.internal_energy),
    });
    const double tolerance =
        256.0 * std::numeric_limits<double>::epsilon();
    if (diagnostics.maximum_relative_invariant_residual > tolerance) {
        throw std::runtime_error(
            "Weighted PIC population control failed invariant closure.");
    }

    diagnostics.applied = true;
    diagnostics.markers_after = refined.size();
    diagnostics.live_markers_after =
        factor * diagnostics.live_markers_before;
    markers = std::move(refined);
    return diagnostics;
}

}  // namespace ev
