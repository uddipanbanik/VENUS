#pragma once

// Convert transported weighted particles into canonical cell-average f(r,w,eta).
// Active particles are assigned to radial cells, then continuous velocity pairs
// are projected onto nonnegative velocity-cell stencils. Each projection
// preserves represented mass, radial momentum and kinetic energy evaluated
// at that destination radial-cell center. Delayed-inner and escaped mass are
// reported separately. An explicitly bounded in-domain continuous carrier can
// retain a packet outside the discrete positive moment hull.
// This synchronization is a representation change; its diagnostics expose the
// projection and carrier budgets rather than asserting an exact trajectory.

#include "ev/reversible_finite_angle_boltzmann.hpp"
#include "ev/weighted_vlasov_pic.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>

namespace ev {

struct WeightedPICPhaseRemapDiagnostics {
    std::size_t active_markers = 0;
    std::size_t projected_pairs = 0;
    std::size_t duplicated_singletons = 0;
    double active_mass = 0.0;
    double hidden_inner_mass = 0.0;
    double escaped_mass = 0.0;
    double radial_momentum_before = 0.0;
    double radial_momentum_after = 0.0;
    double kinetic_energy_before = 0.0;
    double kinetic_energy_after = 0.0;
    double maximum_projection_error = 0.0;
    std::size_t carrier_pairs = 0;
    std::size_t carrier_markers = 0;
    double carrier_mass = 0.0;
    double carrier_radial_momentum = 0.0;
    double carrier_kinetic_energy = 0.0;
};

struct WeightedPICPhaseRemapResult {
    std::vector<double> f;
    std::vector<double> cell_mass;
    std::vector<std::size_t> carrier_marker_indices;
    WeightedPICPhaseRemapDiagnostics diagnostics;
};

struct WeightedPICPhaseRemapOptions {
    // Search remains local.  A value of zero uses twice the configured
    // collision projection radius, with a minimum of four cells.
    int maximum_candidate_radius = 0;
    // Unrepresentable packets remain continuous carriers.  This is a hard
    // representation budget, not a physics tolerance: preflight it on a
    // representative continuous unequal-mass state and audit the carrier
    // ledger. Zero requires every packet to be represented on the grid and
    // rejects a remap that would require a continuous carrier.
    double maximum_carrier_mass_fraction = 0.0;
};

namespace weighted_pic_phase_remap_detail {

inline int containing_cell(double value, const std::vector<double>& edge) {
    if (edge.size() < 2 || value < edge.front() || value > edge.back()) {
        return -1;
    }
    if (value == edge.back()) {
        return static_cast<int>(edge.size()) - 2;
    }
    const auto upper = std::upper_bound(edge.begin(), edge.end(), value);
    if (upper == edge.begin() || upper == edge.end()) {
        return -1;
    }
    return static_cast<int>(upper - edge.begin()) - 1;
}

inline BPFAVec3 reduced_velocity_at_radius(
    const WeightedPICMarker& marker,
    double radius) {
    const double tangential_speed = std::sqrt(2.0 * marker.eta)
        / std::max(radius, std::numeric_limits<double>::min());
    return {marker.w, tangential_speed, 0.0};
}

inline double projection_error(const BPReversibleProjection& projection) {
    return std::max({
        std::abs(projection.mass_error),
        std::abs(projection.momentum_error),
        std::abs(projection.energy_error),
    });
}

}  // namespace weighted_pic_phase_remap_detail

/// Reconstructs cell-average f after marker collisions and/or characteristic
/// transport.  Markers are first assigned to radial cells.  Within each radial
/// cell, markers are grouped by their exact stored mass and sorted by their
/// cell-centred kinetic energy.  The lowest and highest remaining energies in
/// one mass group are paired, with the stable marker identifier breaking
/// ties.  Their two continuous reduced velocities are projected together onto
/// a nonnegative velocity-cell stencil.  A mass group containing only one
/// marker uses the duplicated-singleton construction below.  The projection
/// preserves represented mass, radial momentum, and kinetic energy at the
/// destination radial-cell centre for arbitrary positive marker masses.
///
/// An odd marker is paired with an identical copy and deposited with half the
/// ordinary pair mass.  This keeps the same three invariants without dropping
/// or inventing a marker.  An in-domain packet outside the local nonnegative
/// moment hull can remain as an explicitly indexed continuous carrier, up to
/// the configured mass-fraction bound.  A velocity outside the configured
/// grid is always a hard error.
inline WeightedPICPhaseRemapResult remap_weighted_vlasov_pic_to_phase_space(
    const std::vector<WeightedPICMarker>& markers,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    const WeightedPICPhaseRemapOptions& options) {
    validate_cell_average_eta_geometry(
        geometry, p, "weighted PIC phase-space remap");
    if (options.maximum_candidate_radius < 0
        || !(options.maximum_carrier_mass_fraction >= 0.0)
        || !(options.maximum_carrier_mass_fraction < 1.0)
        || !std::isfinite(options.maximum_carrier_mass_fraction)) {
        throw std::runtime_error("Invalid weighted PIC phase-remap options.");
    }
    const std::size_t cells = static_cast<std::size_t>(p.Nr)
        * static_cast<std::size_t>(p.Nw)
        * static_cast<std::size_t>(p.NL);
    WeightedPICPhaseRemapResult result;
    result.f.assign(cells, 0.0);
    result.cell_mass.assign(cells, 0.0);
    std::vector<double> cell_mass_compensation(cells, 0.0);

    // Separate marker lifecycle state before projection. Hidden-inner and
    // escaped weights remain in their own budgets; only active supported
    // particles enter the Eulerian phase-space reconstruction.
    std::vector<std::vector<std::size_t>> radial_bucket(
        static_cast<std::size_t>(p.Nr));
    weighted_pic_detail::CompensatedMassSum active_mass;
    weighted_pic_detail::CompensatedMassSum hidden_mass;
    weighted_pic_detail::CompensatedMassSum escaped_mass;
    weighted_pic_detail::CompensatedMassSum carrier_mass;
    for (std::size_t marker_index = 0;
         marker_index < markers.size();
         ++marker_index) {
        const WeightedPICMarker& marker = markers[marker_index];
        validate_weighted_pic_marker(marker, "Weighted PIC phase-space remap");
        if (marker.state == WeightedPICMarkerState::delayed_inner) {
            hidden_mass.add(marker.mass);
            continue;
        }
        if (marker.state == WeightedPICMarkerState::escaped) {
            escaped_mass.add(marker.mass);
            continue;
        }
        const int radial_cell = weighted_pic_phase_remap_detail::containing_cell(
            marker.r, geometry.radial.edge);
        if (radial_cell < 0) {
            throw std::runtime_error(
                "An active weighted PIC marker is outside the radial remap domain.");
        }
        radial_bucket[static_cast<std::size_t>(radial_cell)].push_back(
            marker_index);
        ++result.diagnostics.active_markers;
        active_mass.add(static_cast<long double>(marker.mass));
    }
    result.diagnostics.active_mass = static_cast<double>(active_mass.value());
    result.diagnostics.hidden_inner_mass = static_cast<double>(hidden_mass.value());
    result.diagnostics.escaped_mass = static_cast<double>(escaped_mass.value());

    for (int i = 0; i < p.Nr; ++i) {
        std::vector<std::size_t>& bucket =
            radial_bucket[static_cast<std::size_t>(i)];
        const double radius = geometry.radial.center[static_cast<std::size_t>(i)];
        // Group by exact stored mass and then kinetic energy at the destination
        // cell center. Pairing low and high energies broadens the available
        // positive moment stencil while stable ID breaks equal-energy ties.
        std::stable_sort(
            bucket.begin(),
            bucket.end(),
            [&](std::size_t first, std::size_t second) {
                const WeightedPICMarker& first_marker = markers[first];
                const WeightedPICMarker& second_marker = markers[second];
                if (first_marker.mass != second_marker.mass) {
                    return first_marker.mass < second_marker.mass;
                }
                const double first_energy = 0.5 * first_marker.w * first_marker.w
                    + first_marker.eta / (radius * radius);
                const double second_energy = 0.5 * second_marker.w * second_marker.w
                    + second_marker.eta / (radius * radius);
                if (first_energy != second_energy) {
                    return first_energy < second_energy;
                }
                return markers[first].id < markers[second].id;
            });
        auto deposit_pair = [&](std::size_t first_index,
                                std::size_t second_index,
                                double represented_pair_scale) {
            const WeightedPICMarker& first = markers[first_index];
            const WeightedPICMarker& second = markers[second_index];
            const BPFAVec3 first_velocity =
                weighted_pic_phase_remap_detail::reduced_velocity_at_radius(
                    first, radius);
            const BPFAVec3 second_velocity =
                weighted_pic_phase_remap_detail::reduced_velocity_at_radius(
                    second, radius);
            bool outside_domain = false;
            unsigned outside_mask = BP_REVERSIBLE_DOMAIN_NONE;
            BPReversibleProjection projection;
            // Project two continuous reduced velocities jointly. The nonnegative
            // weights must reproduce the pair mass, radial momentum and energy
            // using the same center radius and canonical eta grid.
            bool projected = bp_reversible_project_outgoing_pair(
                    first_velocity,
                    second_velocity,
                    radius,
                    geometry.radial_velocity.center,
                    geometry.eta.center,
                    geometry,
                    p,
                    projection,
                    &outside_domain,
                    &outside_mask);
            if (!projected && !outside_domain) {
                // A continuous pair can need a slightly wider moment hull
                // than one collision event.  Keep this search local; a full
                // velocity-grid search has prohibitive cost and does not fix
                // a point outside the discrete nonnegative moment hull.
                Params fallback = p;
                fallback.bp_reversible_candidate_radius =
                    options.maximum_candidate_radius > 0
                    ? options.maximum_candidate_radius
                    : std::min(
                        std::max(p.Nw, p.NL),
                        std::max(4, 2 * p.bp_reversible_candidate_radius));
                if (fallback.bp_reversible_candidate_radius
                    > p.bp_reversible_candidate_radius) {
                    projected = bp_reversible_project_outgoing_pair(
                        first_velocity,
                        second_velocity,
                        radius,
                        geometry.radial_velocity.center,
                        geometry.eta.center,
                        geometry,
                        fallback,
                        projection,
                        &outside_domain,
                        &outside_mask);
                }
            }
            // An out-of-domain velocity is a hard domain failure. An in-domain
            // state outside the local positive moment hull can remain continuous
            // only within the explicitly configured carrier mass budget.
            if (!projected) {
                const double packet_mass = represented_pair_scale * first.mass;
                const double carried_mass = 2.0 * packet_mass;
                const double proposed_carrier_mass =
                    static_cast<double>(carrier_mass.value() + carried_mass);
                const double carrier_fraction = result.diagnostics.active_mass > 0.0
                    ? proposed_carrier_mass / result.diagnostics.active_mass
                    : 0.0;
                if (!outside_domain
                    && options.maximum_carrier_mass_fraction > 0.0
                    && carrier_fraction
                        <= options.maximum_carrier_mass_fraction
                            + 64.0 * std::numeric_limits<double>::epsilon()) {
                    ++result.diagnostics.carrier_pairs;
                    carrier_mass.add(carried_mass);
                    result.diagnostics.carrier_mass =
                        static_cast<double>(carrier_mass.value());
                    result.diagnostics.carrier_radial_momentum += packet_mass
                        * (first_velocity.x + second_velocity.x);
                    result.diagnostics.carrier_kinetic_energy +=
                        0.5 * packet_mass
                        * (bp_fa_dot(first_velocity, first_velocity)
                           + bp_fa_dot(second_velocity, second_velocity));
                    result.carrier_marker_indices.push_back(first_index);
                    if (second_index != first_index) {
                        result.carrier_marker_indices.push_back(second_index);
                    }
                    result.diagnostics.carrier_markers =
                        result.carrier_marker_indices.size();
                    return false;
                }
                std::ostringstream message;
                message
                    << "Weighted PIC phase remap could not represent an active marker pair"
                    << " in radial cell " << i
                    << "; outside_domain=" << (outside_domain ? 1 : 0)
                    << ", domain_mask=" << outside_mask
                    << ", v1=(" << first_velocity.x << ','
                    << first_velocity.y << ',' << first_velocity.z << ')'
                    << ", v2=(" << second_velocity.x << ','
                    << second_velocity.y << ',' << second_velocity.z << ')'
                    << ", proposed_carrier_fraction=" << carrier_fraction
                    << ". Enlarge w_max/L_max when domain_mask is nonzero; "
                    << "otherwise refine the local velocity grid or enable "
                    << "a bounded continuous-carrier fraction.";
                throw std::runtime_error(message.str());
            }
            const double packet_mass = represented_pair_scale * first.mass;
            // Accumulate represented mass first; dividing by the physical phase-
            // space measure occurs only after every pair contribution has arrived.
            // Compensation retains small deposits beside much larger ones.
            auto add_cell_mass = [&](std::size_t flat, double deposited) {
                // Neumaier summation retains the small deposits that would
                // otherwise be rounded away in a heavily occupied cell.
                const double sum = result.cell_mass[flat];
                const double updated = sum + deposited;
                cell_mass_compensation[flat] += std::abs(sum) >= std::abs(deposited)
                    ? (sum - updated) + deposited
                    : (deposited - updated) + sum;
                result.cell_mass[flat] = updated;
            };
            for (const BPReversibleProductPair& product : projection.products) {
                const std::array<int, 2> first_indices =
                    bp_reversible_velocity_indices(product.first, p);
                const std::array<int, 2> second_indices =
                    bp_reversible_velocity_indices(product.second, p);
                const double deposited = packet_mass * product.weight;
                add_cell_mass(
                    idx(i, first_indices[0], first_indices[1], p), deposited);
                add_cell_mass(
                    idx(i, second_indices[0], second_indices[1], p), deposited);
            }
            ++result.diagnostics.projected_pairs;
            result.diagnostics.maximum_projection_error = std::max(
                result.diagnostics.maximum_projection_error,
                weighted_pic_phase_remap_detail::projection_error(projection));
            return true;
        };

        // Equal masses are required only inside one two-velocity projection,
        // because the existing nonnegative stencil represents two equal
        // packets.  Arbitrary marker weights are supported by forming exact
        // mass groups; a unique weight is projected as one duplicated
        // singleton.  No marker is split or changed.
        for (std::size_t group_begin = 0;
             group_begin < bucket.size();) {
            std::size_t group_end = group_begin + 1;
            while (group_end < bucket.size()
                   && markers[bucket[group_end]].mass
                       == markers[bucket[group_begin]].mass) {
                ++group_end;
            }
            const std::size_t group_size = group_end - group_begin;
            const std::size_t pair_count = group_size / 2;
            for (std::size_t q = 0; q < pair_count; ++q) {
                const std::size_t first_index = bucket[group_begin + q];
                const std::size_t second_index =
                    bucket[group_end - 1 - q];
                const WeightedPICMarker& first = markers[first_index];
                const WeightedPICMarker& second = markers[second_index];
                const BPFAVec3 first_velocity =
                    weighted_pic_phase_remap_detail::reduced_velocity_at_radius(
                        first, radius);
                const BPFAVec3 second_velocity =
                    weighted_pic_phase_remap_detail::reduced_velocity_at_radius(
                        second, radius);
                result.diagnostics.radial_momentum_before +=
                    first.mass
                    * (first_velocity.x + second_velocity.x);
                result.diagnostics.kinetic_energy_before +=
                    0.5 * first.mass
                    * (bp_fa_dot(first_velocity, first_velocity)
                       + bp_fa_dot(second_velocity, second_velocity));
                (void)deposit_pair(first_index, second_index, 1.0);
            }
            // Project an odd singleton as two identical half-weight copies. This
            // uses the same two-velocity moment solver without dropping a sample
            // or adding represented mass.
            if ((group_size & 1U) != 0U) {
                const std::size_t marker_index =
                    bucket[group_begin + pair_count];
                const WeightedPICMarker& marker = markers[marker_index];
                const BPFAVec3 velocity =
                    weighted_pic_phase_remap_detail::reduced_velocity_at_radius(
                        marker, radius);
                result.diagnostics.radial_momentum_before +=
                    marker.mass * velocity.x;
                result.diagnostics.kinetic_energy_before +=
                    0.5 * marker.mass * bp_fa_dot(velocity, velocity);
                (void)deposit_pair(marker_index, marker_index, 0.5);
                ++result.diagnostics.duplicated_singletons;
            }
            group_begin = group_end;
        }
    }

    for (std::size_t flat = 0; flat < cells; ++flat) {
        result.cell_mass[flat] += cell_mass_compensation[flat];
    }
    result.diagnostics.radial_momentum_after +=
        result.diagnostics.carrier_radial_momentum;
    result.diagnostics.kinetic_energy_after +=
        result.diagnostics.carrier_kinetic_energy;

    for (int i = 0; i < p.Nr; ++i) {
        const double radius = geometry.radial.center[static_cast<std::size_t>(i)];
        for (int j = 0; j < p.Nw; ++j) {
            const double radial_velocity =
                geometry.radial_velocity.center[static_cast<std::size_t>(j)];
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t flat = idx(i, j, k, p);
                const double mass = result.cell_mass[flat];
                const double energy = 0.5 * radial_velocity * radial_velocity
                    + geometry.eta.center[static_cast<std::size_t>(k)]
                        / (radius * radius);
                result.diagnostics.radial_momentum_after +=
                    mass * radial_velocity;
                result.diagnostics.kinetic_energy_after += mass * energy;
                const double measure = 8.0 * PI * PI
                    * geometry.radial.width[static_cast<std::size_t>(i)]
                    * geometry.radial_velocity.width[static_cast<std::size_t>(j)]
                    * geometry.eta.width[static_cast<std::size_t>(k)];
                // The canonical spherical cell measure is 8pi^2 dr dw deta. Its ratio
                // with the deposited cell mass defines the returned cell-average f.
                result.f[flat] = mass / measure;
            }
        }
    }

    weighted_pic_detail::CompensatedMassSum deposited_mass_sum;
    for (double mass : result.cell_mass) {
        deposited_mass_sum.add(static_cast<long double>(mass));
    }
    const double deposited_mass = static_cast<double>(deposited_mass_sum.value());
    // The total active budget includes both grid-deposited mass and
    // explicit continuous carriers. Hidden and escaped weights were
    // separately recorded, so they cannot hide an active remap deficit.
    const double mass_tolerance = 256.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, result.diagnostics.active_mass);
    if (std::abs(
            deposited_mass + result.diagnostics.carrier_mass
            - result.diagnostics.active_mass)
        > mass_tolerance) {
        std::ostringstream message;
        message.precision(17);
        message << "Weighted PIC phase remap failed its active-mass budget: before="
                << result.diagnostics.active_mass
                << ", grid_after=" << deposited_mass
                << ", carrier_after=" << result.diagnostics.carrier_mass
                << ", difference="
                << deposited_mass + result.diagnostics.carrier_mass
                    - result.diagnostics.active_mass
                << ", tolerance=" << mass_tolerance << ".";
        throw std::runtime_error(message.str());
    }
    return result;
}

inline WeightedPICPhaseRemapResult remap_weighted_vlasov_pic_to_phase_space(
    const std::vector<WeightedPICMarker>& markers,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    return remap_weighted_vlasov_pic_to_phase_space(
        markers, geometry, p, WeightedPICPhaseRemapOptions{});
}

}  // namespace ev
