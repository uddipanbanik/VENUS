#pragma once

// Represent finite-volume mass temporarily inside the unresolved central sphere.
// The carrier is a persistent collection of numerical mass packets whose
// physical return times determine when they rejoin the resolved distribution.
// Inputs are packets with reduced mass, an eta quadrature value, inward boundary
// speed, and remaining return time, together with the phase-space geometry.
// This header validates those records, evaluates physical mass/moment budgets,
// compresses compatible packet groups, and computes uniform-core flight times.
// eta=L^2/2 retains each packet's actual angular momentum while its radial
// path is hidden below r_min. The surrounding transport advances the clocks
// and returns their mass to the resolved grid at the prescribed boundary time.

#include "ev/phase_space_cells.hpp"

#include <unordered_map>

namespace ev {

/// One finite-volume mass packet currently inside 0 < r < r_min.
struct DelayedInnerCarrierPacket {
    // Integral of f dr dw in one eta cell.  Multiplication by
    // 8*pi^2*deta gives physical mass.
    double reduced_mass = 0.0;
    int eta_index = -1;
    // The quadrature abscissa inside the eta cell.  Keeping this value avoids
    // replacing a low-angular-momentum orbit by the cell-centre orbit while
    // the packet is hidden inside r_min.
    double eta = 0.0;
    double boundary_speed = 0.0;
    double remaining_time = 0.0;
};

/// Persistent hidden state associated with one local eta slab.
struct DelayedInnerCarrier {
    std::vector<DelayedInnerCarrierPacket> packets;
};

struct DelayedInnerCarrierLedger {
    double physical_mass = 0.0;
    double reduced_mass = 0.0;
    double minimum_remaining_time = std::numeric_limits<double>::infinity();
    double maximum_remaining_time = 0.0;
    double maximum_boundary_speed = 0.0;
    std::size_t packet_count = 0;
};

// Compare compression through physical mass, outward boundary momentum,
// kinetic energy including eta/r_min^2, and mass-weighted absolute return time.
struct DelayedInnerCarrierMomentLedger {
    double physical_mass = 0.0;
    double boundary_radial_momentum = 0.0;
    double boundary_kinetic_energy = 0.0;
    double absolute_return_time_moment = 0.0;
    std::size_t packet_count = 0;
};

struct DelayedInnerCarrierCompressionLedger {
    bool applied = false;
    DelayedInnerCarrierMomentLedger before;
    DelayedInnerCarrierMomentLedger after;
};

// Reduced mass must be positive and orbit/clock fields finite. The eta-cell
// index identifies the quadrature slab needed to recover physical mass.
inline void validate_delayed_inner_carrier_packet(
    const DelayedInnerCarrierPacket& packet,
    int eta_cells,
    const char* label) {
    if (!(packet.reduced_mass > 0.0)
        || !std::isfinite(packet.reduced_mass)
        || packet.eta_index < 0
        || packet.eta_index >= eta_cells
        || !(packet.eta >= 0.0)
        || !std::isfinite(packet.eta)
        || !(packet.boundary_speed >= 0.0)
        || !std::isfinite(packet.boundary_speed)
        || !(packet.remaining_time >= 0.0)
        || !std::isfinite(packet.remaining_time)) {
        throw std::runtime_error(
            std::string(label) + " contains an invalid delayed-return packet.");
    }
}

// Restore spherical phase volume: dM=8*pi^2*deta*(reduced packet mass).
// Check that the retained eta abscissa remains inside its own cell.
inline DelayedInnerCarrierLedger evaluate_delayed_inner_carrier_ledger(
    const DelayedInnerCarrier& carrier,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    if (geometry.eta.width.size() != static_cast<std::size_t>(p.NL)) {
        throw std::runtime_error(
            "Delayed-return carrier eta geometry does not match the local grid.");
    }
    DelayedInnerCarrierLedger ledger;
    constexpr double phase_factor = 8.0 * PI * PI;
    for (const DelayedInnerCarrierPacket& packet : carrier.packets) {
        validate_delayed_inner_carrier_packet(
            packet, p.NL, "Delayed-return carrier");
        const double physical_mass = phase_factor
            * geometry.eta.width[static_cast<std::size_t>(packet.eta_index)]
            * packet.reduced_mass;
        const std::size_t k = static_cast<std::size_t>(packet.eta_index);
        const double eta_scale = std::max(1.0, geometry.eta.edge[k + 1]);
        const double eta_tolerance = 128.0
            * std::numeric_limits<double>::epsilon() * eta_scale;
        if (packet.eta < geometry.eta.edge[k] - eta_tolerance
            || packet.eta > geometry.eta.edge[k + 1] + eta_tolerance) {
            throw std::runtime_error(
                "Delayed-return carrier packet eta lies outside its cell.");
        }
        ledger.physical_mass += physical_mass;
        ledger.reduced_mass += packet.reduced_mass;
        ledger.minimum_remaining_time = std::min(
            ledger.minimum_remaining_time, packet.remaining_time);
        ledger.maximum_remaining_time = std::max(
            ledger.maximum_remaining_time, packet.remaining_time);
        ledger.maximum_boundary_speed = std::max(
            ledger.maximum_boundary_speed, packet.boundary_speed);
    }
    ledger.packet_count = carrier.packets.size();
    return ledger;
}

// Boundary speed is an outward return speed. Include tangential kinetic
// energy at r_min, and use current_time+remaining_time to compare fixed
// physical return epochs while the carrier clock advances.
inline DelayedInnerCarrierMomentLedger
evaluate_delayed_inner_carrier_moment_ledger(
    const DelayedInnerCarrier& carrier,
    const PhaseSpaceCellGeometry& geometry,
    double current_time,
    const Params& p) {
    if (!(current_time >= 0.0) || !std::isfinite(current_time)) {
        throw std::runtime_error(
            "Delayed-return carrier moment ledger received invalid time.");
    }
    DelayedInnerCarrierMomentLedger ledger;
    constexpr long double phase_factor =
        8.0L * static_cast<long double>(PI) * static_cast<long double>(PI);
    long double mass = 0.0L;
    long double momentum = 0.0L;
    long double kinetic_energy = 0.0L;
    long double return_time_moment = 0.0L;
    for (const DelayedInnerCarrierPacket& packet : carrier.packets) {
        validate_delayed_inner_carrier_packet(
            packet, p.NL, "Delayed-return carrier moment ledger");
        const std::size_t k = static_cast<std::size_t>(packet.eta_index);
        const long double physical_mass = phase_factor
            * static_cast<long double>(geometry.eta.width[k])
            * static_cast<long double>(packet.reduced_mass);
        const long double speed = packet.boundary_speed;
        const long double boundary_energy =
            0.5L * speed * speed
            + static_cast<long double>(packet.eta)
                / (static_cast<long double>(p.r_min) * p.r_min);
        const long double absolute_return_time =
            static_cast<long double>(current_time)
            + packet.remaining_time;
        mass += physical_mass;
        momentum += physical_mass * speed;
        kinetic_energy += physical_mass * boundary_energy;
        return_time_moment += physical_mass * absolute_return_time;
    }
    ledger.physical_mass = static_cast<double>(mass);
    ledger.boundary_radial_momentum = static_cast<double>(momentum);
    ledger.boundary_kinetic_energy = static_cast<double>(kinetic_energy);
    ledger.absolute_return_time_moment =
        static_cast<double>(return_time_moment);
    ledger.packet_count = carrier.packets.size();
    return ledger;
}

// Group only a common eta cell/abscissa, boundary-speed cell, and absolute
// return-time bin. Exact eta protects the unresolved orbit's turning point.
struct DelayedInnerCarrierCompressionKey {
    int eta_index = 0;
    double eta = 0.0;
    int speed_bin = 0;
    std::int64_t return_time_bin = 0;

    bool operator==(const DelayedInnerCarrierCompressionKey& other) const {
        return eta_index == other.eta_index
            && eta == other.eta
            && speed_bin == other.speed_bin
            && return_time_bin == other.return_time_bin;
    }
};

struct DelayedInnerCarrierCompressionKeyHash {
    std::size_t operator()(
        const DelayedInnerCarrierCompressionKey& key) const noexcept {
        std::size_t seed = std::hash<std::int64_t>{}(key.return_time_bin);
        seed ^= std::hash<int>{}(key.eta_index)
            + static_cast<std::size_t>(0x9e3779b9U)
            + (seed << 6U) + (seed >> 2U);
        seed ^= std::hash<double>{}(key.eta)
            + static_cast<std::size_t>(0x9e3779b9U)
            + (seed << 6U) + (seed >> 2U);
        seed ^= std::hash<int>{}(key.speed_bin)
            + static_cast<std::size_t>(0x9e3779b9U)
            + (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

struct DelayedInnerCarrierCompressionAccumulator {
    long double reduced_mass = 0.0L;
    long double mass_speed = 0.0L;
    long double mass_speed2 = 0.0L;
    long double mass_return_time = 0.0L;
    double minimum_speed = std::numeric_limits<double>::infinity();
    double maximum_speed = 0.0;
};

/// Coalesces hidden packets only within one eta cell, one resolved boundary-
/// speed cell, and one absolute-return-time bin.  Each group retains its mass,
/// mean boundary speed, boundary-speed second moment, and mean return time.
/// The three speed nodes are the unique nonnegative min/mean/max quadrature
/// for those moments.  Binning absolute return time prevents repeated calls
/// from adding secular timing diffusion as the carrier clock advances.
inline DelayedInnerCarrierCompressionLedger
compress_delayed_inner_carrier(
    DelayedInnerCarrier& carrier,
    const PhaseSpaceCellGeometry& geometry,
    double current_time,
    const Params& p) {
    DelayedInnerCarrierCompressionLedger ledger;
    ledger.before = evaluate_delayed_inner_carrier_moment_ledger(
        carrier, geometry, current_time, p);
    // Compression is an optional reduction of representation size. A zero
    // bin width or a small packet population retains the original packets.
    const double bin_width = p.vp_inner_carrier_return_time_bin_width;
    if (!(bin_width > 0.0)
        || carrier.packets.size() < static_cast<std::size_t>(
            p.vp_inner_carrier_compress_min_packets)) {
        ledger.after = ledger.before;
        return ledger;
    }

    using Key = DelayedInnerCarrierCompressionKey;
    using Accumulator = DelayedInnerCarrierCompressionAccumulator;
    std::unordered_map<Key, Accumulator,
        DelayedInnerCarrierCompressionKeyHash> group;
    group.reserve(carrier.packets.size() / 4 + 1);
    std::vector<DelayedInnerCarrierPacket> imminent;
    imminent.reserve(carrier.packets.size() / 16 + 1);
    const std::vector<double>& speed_edge = geometry.radial_velocity.edge;
    for (const DelayedInnerCarrierPacket& packet : carrier.packets) {
        validate_delayed_inner_carrier_packet(
            packet, p.NL, "Delayed-return carrier compression input");
        // Keep imminent returns individually so compression cannot move
        // their mass across the next short boundary-release interval.
        if (packet.remaining_time <= bin_width) {
            imminent.push_back(packet);
            continue;
        }
        int speed_bin = 0;
        if (packet.boundary_speed > speed_edge.back()) {
            speed_bin = p.Nw;
        } else if (packet.boundary_speed < speed_edge.front()) {
            speed_bin = -1;
        } else {
            speed_bin = static_cast<int>(
                std::upper_bound(
                    speed_edge.begin(), speed_edge.end(),
                    packet.boundary_speed)
                - speed_edge.begin()) - 1;
            speed_bin = std::clamp(speed_bin, 0, p.Nw - 1);
        }
        const long double absolute_return_time =
            static_cast<long double>(current_time)
            + packet.remaining_time;
        // Bin absolute physical return epochs, not the remaining delay.
        // This keeps the grouping fixed as current_time advances.
        const long double scaled_time =
            absolute_return_time / static_cast<long double>(bin_width);
        if (!(scaled_time >= 0.0L)
            || scaled_time
                > static_cast<long double>(
                    std::numeric_limits<std::int64_t>::max())) {
            throw std::runtime_error(
                "Delayed-return carrier return-time bin overflow.");
        }
        const Key key{
            packet.eta_index,
            packet.eta,
            speed_bin,
            static_cast<std::int64_t>(std::floor(scaled_time)),
        };
        Accumulator& value = group[key];
        const long double packet_mass = packet.reduced_mass;
        const long double speed = packet.boundary_speed;
        value.reduced_mass += packet_mass;
        value.mass_speed += packet_mass * speed;
        value.mass_speed2 += packet_mass * speed * speed;
        value.mass_return_time += packet_mass * absolute_return_time;
        value.minimum_speed = std::min(
            value.minimum_speed, packet.boundary_speed);
        value.maximum_speed = std::max(
            value.maximum_speed, packet.boundary_speed);
    }

    // Hash accumulation groups compatible packets; sort the groups before
    // publication to give the compressed carrier a deterministic order.
    std::vector<std::pair<Key, Accumulator>> ordered;
    ordered.reserve(group.size());
    for (const auto& item : group) {
        ordered.push_back(item);
    }
    std::sort(
        ordered.begin(), ordered.end(),
        [](const auto& first, const auto& second) {
            if (first.first.eta_index != second.first.eta_index) {
                return first.first.eta_index < second.first.eta_index;
            }
            if (first.first.eta != second.first.eta) {
                return first.first.eta < second.first.eta;
            }
            if (first.first.return_time_bin
                != second.first.return_time_bin) {
                return first.first.return_time_bin
                    < second.first.return_time_bin;
            }
            return first.first.speed_bin < second.first.speed_bin;
        });

    std::vector<DelayedInnerCarrierPacket> compressed;
    compressed.reserve(imminent.size() + 3 * ordered.size());
    compressed.insert(
        compressed.end(),
        std::make_move_iterator(imminent.begin()),
        std::make_move_iterator(imminent.end()));
    // Convert a positive quadrature weight back to a packet and its
    // remaining delay relative to the same current physical time.
    auto append_packet = [&](long double reduced_mass,
                             int eta_index,
                             double eta,
                             long double speed,
                             long double absolute_return_time) {
        const double packet_mass = static_cast<double>(reduced_mass);
        if (!(packet_mass > 0.0)) {
            return;
        }
        compressed.push_back({
            packet_mass,
            eta_index,
            eta,
            static_cast<double>(speed),
            std::max(
                0.0,
                static_cast<double>(
                    absolute_return_time
                    - static_cast<long double>(current_time))),
        });
    };
    for (const auto& item : ordered) {
        const Key& key = item.first;
        const Accumulator& value = item.second;
        const long double mass = value.reduced_mass;
        if (!(mass > 0.0L)) {
            continue;
        }
        const long double mean_speed = value.mass_speed / mass;
        const long double mean_speed2 = value.mass_speed2 / mass;
        const long double mean_return_time =
            value.mass_return_time / mass;
        const long double lower_span = std::max(
            0.0L,
            mean_speed - static_cast<long double>(value.minimum_speed));
        const long double upper_span = std::max(
            0.0L,
            static_cast<long double>(value.maximum_speed) - mean_speed);
        const long double moment_scale = std::max({
            1.0e-300L,
            std::abs(mean_speed2),
            mean_speed * mean_speed,
        });
        // Infer the boundary-speed variance from extensive moments. Its
        // admissible range is limited by the actual minimum/maximum speeds.
        long double variance = mean_speed2 - mean_speed * mean_speed;
        if (variance < 0.0L
            && std::abs(variance)
                <= 256.0L * std::numeric_limits<long double>::epsilon()
                    * moment_scale) {
            variance = 0.0L;
        }
        const long double maximum_variance = lower_span * upper_span;
        variance = std::clamp(variance, 0.0L, maximum_variance);
        const long double degenerate_tolerance =
            256.0L * std::numeric_limits<long double>::epsilon()
            * moment_scale;
        if (!(variance > degenerate_tolerance)
            || !(lower_span > 0.0L)
            || !(upper_span > 0.0L)) {
            append_packet(
                mass, key.eta_index, key.eta, mean_speed, mean_return_time);
            continue;
        }
        const long double span = lower_span + upper_span;
        // Weights on minimum, mean, and maximum speed retain mass, mean
        // momentum, and kinetic second moment with nonnegative packet masses.
        const long double lower_mass =
            mass * variance / (lower_span * span);
        const long double upper_mass =
            mass * variance / (upper_span * span);
        const long double center_mass = std::max(
            0.0L, mass - lower_mass - upper_mass);
        append_packet(
            lower_mass,
            key.eta_index,
            key.eta,
            value.minimum_speed,
            mean_return_time);
        append_packet(
            center_mass,
            key.eta_index,
            key.eta,
            mean_speed,
            mean_return_time);
        append_packet(
            upper_mass,
            key.eta_index,
            key.eta,
            value.maximum_speed,
            mean_return_time);
    }
    carrier.packets = std::move(compressed);
    ledger.applied = true;
    ledger.after = evaluate_delayed_inner_carrier_moment_ledger(
        carrier, geometry, current_time, p);

    // Recompute moments from the installed packet representation and reject
    // compression whose rounding error exceeds the declared relative bound.
    auto require_moment = [](double before, double after, const char* label) {
        const double scale = std::max(std::abs(before), 1.0e-300);
        if (std::abs(after - before) > 2.0e-12 * scale) {
            std::ostringstream message;
            message
                << "Delayed-return carrier compression changed " << label
                << ": before=" << before << ", after=" << after;
            throw std::runtime_error(message.str());
        }
    };
    require_moment(
        ledger.before.physical_mass,
        ledger.after.physical_mass,
        "mass");
    require_moment(
        ledger.before.boundary_radial_momentum,
        ledger.after.boundary_radial_momentum,
        "boundary radial momentum");
    require_moment(
        ledger.before.boundary_kinetic_energy,
        ledger.after.boundary_kinetic_energy,
        "boundary kinetic energy");
    require_moment(
        ledger.before.absolute_return_time_moment,
        ledger.after.absolute_return_time_moment,
        "absolute return-time moment");
    // Bound memory even after moment-preserving compression; an unresolved
    // carrier that still exceeds the limit requires a different resolution.
    if (carrier.packets.size()
        > static_cast<std::size_t>(p.vp_inner_carrier_max_packets)) {
        throw std::runtime_error(
            "Compressed delayed-return carrier still exceeds "
            "vp_inner_carrier_max_packets.");
    }
    return ledger;
}

/// Returns the exact round-trip time in a uniform-density central cell.
///
/// The unresolved carrier mass M produces
///
///   Phi(r)-Phi(R) = G M (r^2-R^2)/(2 R^3),  R=r_min.
///
/// With boundary radial speed v and angular momentum L, y=r^2 obeys
///
///   (dy/dt)^2 = 4[-C y^2 + A y - L^2],
///   C=G M/R^3,
///   A=v^2+L^2/R^2+C R^2.
///
/// This quadratic gives the pericentre and flight time analytically.  The
/// C->0 expression is evaluated separately to avoid cancellation.
inline double delayed_inner_uniform_cell_round_trip_time(
    double boundary_speed,
    double angular_momentum,
    double hidden_mass,
    const Params& p) {
    if (!(p.r_min > 0.0)
        || !(boundary_speed >= 0.0)
        || !std::isfinite(boundary_speed)
        || !(angular_momentum >= 0.0)
        || !std::isfinite(angular_momentum)
        || !(hidden_mass >= 0.0)
        || !std::isfinite(hidden_mass)) {
        throw std::runtime_error(
            "Delayed-return flight time received invalid orbit data.");
    }
    if (p.central_mass != 0.0) {
        throw std::runtime_error(
            "The delayed-return central-cell model currently requires "
            "central_mass=0; a point-mass centre needs a separate regularized "
            "interior potential.");
    }

    const long double R = p.r_min;
    const long double v = boundary_speed;
    const long double L = angular_momentum;
    const long double B = L * L;
    const long double C = static_cast<long double>(p.vp_gravity)
        * hidden_mass / (R * R * R);
    const long double A = v * v + B / (R * R) + C * R * R;
    if (!(A > 0.0L)) {
        // A zero-speed radial orbit at the face has zero measure in the flux.
        return 0.0;
    }

    // Compare the uniform-core harmonic term with the orbit energy scale.
    // Its free-flight limit avoids subtracting nearly equal turning roots.
    const long double small_C = 64.0L
        * std::numeric_limits<long double>::epsilon()
        * A / (R * R);
    if (!(C > small_C)) {
        const long double y_peri = B / A;
        const long double radial_span = std::max(0.0L, R * R - y_peri);
        return static_cast<double>(
            2.0L * std::sqrt(radial_span / A));
    }

    long double discriminant = A * A - 4.0L * C * B;
    const long double discriminant_scale = std::max(A * A, 1.0L);
    if (discriminant < 0.0L
        && std::abs(discriminant)
            <= 256.0L * std::numeric_limits<long double>::epsilon()
                * discriminant_scale) {
        discriminant = 0.0L;
    }
    if (!(discriminant >= 0.0L)) {
        throw std::runtime_error(
            "Delayed-return central-cell orbit has no real radial turning point.");
    }
    const long double root = std::sqrt(discriminant);
    const long double y_plus = (A + root) / (2.0L * C);
    // Rationalize the smaller root to retain a low-angular-momentum
    // pericentre when A and sqrt(discriminant) are nearly equal.
    const long double y_minus = B > 0.0L
        ? 2.0L * B / (A + root)
        : 0.0L;
    const long double half_span = 0.5L * (y_plus - y_minus);
    if (!(half_span > 0.0L)) {
        return 0.0;
    }
    const long double midpoint = 0.5L * (y_plus + y_minus);
    // Integrate between the boundary radius and pericentre on both inward
    // and outward legs; the inverse cosine gives that complete return phase.
    const long double cosine = std::clamp(
        (midpoint - R * R) / half_span, -1.0L, 1.0L);
    const long double theta = std::acos(cosine);
    const long double round_trip = theta / std::sqrt(C);
    if (!(round_trip >= 0.0L) || !std::isfinite(round_trip)) {
        throw std::runtime_error(
            "Delayed-return central-cell flight time is non-finite.");
    }
    return static_cast<double>(round_trip);
}

}  // namespace ev
