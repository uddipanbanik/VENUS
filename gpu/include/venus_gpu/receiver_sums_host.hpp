#pragma once

// Turn device receiver sums into an optional field-variation timescale.
// Inputs are block partials, resolved shell masses/effective counts, and two
// potential/force observations separated by dt. Outputs are a mass-weighted
// relative change, its sampling floor, and a timescale. Fixed-order host sums
// complete the device reduction. Explicit physical and hidden-mass checks
// establish that the receiver measure covers the same particles as gravity.

#include "receiver_sums_math.hpp"
#include "ev/weighted_pic_mass_rms_field_clock.hpp"
#include <algorithm>
#include <limits>
#include <vector>

namespace venus_gpu_receiver {
struct ReceiverSummary {
    long double physical_mass = 0, physical_mass2 = 0, hidden_mass = 0;
    long double phi0_square = 0, phi1_square = 0, phi_difference_square = 0;
    long double force0_square = 0, force1_square = 0, force_difference_square = 0;
    std::uint64_t active_count = 0, hidden_count = 0;
};

inline void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Sum blocks in their stored order using long double host accumulators.
// Validate every partial before using it in the final normalized clock.
inline ReceiverSummary finish_partials(const std::vector<ReceiverTerms>& blocks,
                                      std::size_t expected_active) {
    ReceiverSummary result;
    for (const auto& block : blocks) {
        const double values[] = {block.physical_mass, block.physical_mass2,
            block.hidden_mass, block.phi0_square, block.phi1_square,
            block.phi_difference_square, block.force0_square,
            block.force1_square, block.force_difference_square};
        for (double value : values)
            require(std::isfinite(value) && value >= 0,
                    "Invalid device receiver block partial.");
        result.physical_mass += block.physical_mass;
        result.physical_mass2 += block.physical_mass2;
        result.hidden_mass += block.hidden_mass;
        result.phi0_square += block.phi0_square;
        result.phi1_square += block.phi1_square;
        result.phi_difference_square += block.phi_difference_square;
        result.force0_square += block.force0_square;
        result.force1_square += block.force1_square;
        result.force_difference_square += block.force_difference_square;
        result.active_count += block.active_count;
        result.hidden_count += block.hidden_count;
    }
    // A missing or duplicated active receiver would change the measure
    // even when its normalized field ratio happened to remain finite.
    require(result.active_count == expected_active
        && result.hidden_count <= result.active_count
        && result.physical_mass > 0 && result.physical_mass2 > 0,
        "Device receiver population or physical mass is incomplete.");
    return result;
}

// Keep the selected timescale together with the mass residuals and an
// explicit indication that the caller's maximum-norm clock was used.
struct MixedClock {
    ev::weighted_pic_field_clock::Measurement measured;
    ReceiverSummary receiver;
    bool maximum_fallback = false;
    long double hidden_mass_residual = 0;
    long double total_mass_residual = 0;
};

inline MixedClock finish_clock(
        const std::vector<double>& phi0, const std::vector<double>& force0,
        const std::vector<double>& phi1, const std::vector<double>& force1,
        const std::vector<double>& mass, const std::vector<double>& neff,
        double dt, double expected_hidden, double literal_maximum_time,
        const ReceiverSummary& receiver, bool complete) {
    using ev::weighted_pic_field_clock::mass_rms_change_clock;
    require(std::isfinite(expected_hidden) && expected_hidden >= 0
        && literal_maximum_time > 0,
        "Invalid hidden mass or original maximum-norm clock.");
    MixedClock result;
    result.receiver = receiver;
    // A complete receiver measure is required to replace the supplied
    // maximum-norm field clock with the mass-weighted observation.
    if (!complete) {
        result.maximum_fallback = true;
        result.measured.timescale = literal_maximum_time;
        return result;
    }
    result.measured = mass_rms_change_clock(
        phi0, force0, phi1, force1, mass, neff, dt);
    // Resolved shell mass and hidden cloud mass partition the physical
    // receiver population. Verify that partition before mixing field moments.
    long double represented = 0;
    for (double m : mass) represented += m;
    represented += expected_hidden;
    result.total_mass_residual = receiver.physical_mass - represented;
    require(std::abs(result.total_mass_residual)
        <= 2e-11L * std::max(receiver.physical_mass, 1e-300L),
        "Physical receiver mass does not close the field source ledger.");
    result.hidden_mass_residual = receiver.hidden_mass - expected_hidden;
    require(std::abs(result.hidden_mass_residual)
        <= 2048.L * std::numeric_limits<double>::epsilon()
            * std::max(static_cast<long double>(expected_hidden), 1e-300L),
        "Hidden receiver quadrature does not close deposited hidden mass.");
    // With no hidden source the resolved measurement is already complete.
    if (expected_hidden == 0) {
        require(receiver.hidden_count == 0,
            "Unexpected positive hidden receiver with zero source mass.");
        return result;
    }
    if (!(result.measured.receiver_mass > 0)) {
        result.maximum_fallback = true;
        result.measured.timescale = literal_maximum_time;
        return result;
    }
    // Build quadratic norms for both epochs and their difference, using
    // current receiver weights for the pair of field observations.
    long double p0 = 0, p1 = 0, dp = 0, g0 = 0, g1 = 0, dg = 0;
    for (std::size_t i = 0; i < mass.size(); ++i) {
        const long double m = mass[i], a = phi0[i], b = phi1[i];
        const long double c = force0[i], d = force1[i];
        p0 += m*a*a; p1 += m*b*b; dp += m*(b-a)*(b-a);
        g0 += m*c*c; g1 += m*d*d; dg += m*(d-c)*(d-c);
    }
    p0 += receiver.phi0_square; p1 += receiver.phi1_square;
    dp += receiver.phi_difference_square;
    g0 += receiver.force0_square; g1 += receiver.force1_square;
    dg += receiver.force_difference_square;
    // Normalize by the larger endpoint norm so a field approaching zero
    // remains a finite fractional observation. Zero norm gives zero change.
    const auto relative = [](long double a, long double b, long double d) {
        const long double norm = std::max(a,b);
        return norm > 0 ? static_cast<double>(std::sqrt(d/norm)) : 0.;
    };
    result.measured.receiver_mass += receiver.hidden_mass;
    result.measured.potential_relative_change = relative(p0,p1,dp);
    result.measured.force_relative_change = relative(g0,g1,dg);
    result.measured.potential_signal_resolved = false;
    result.measured.force_signal_resolved = false;
    // Remove the existing sampling floor in quadrature. Changes below that
    // floor carry no resolved variation rate and therefore no finite clock.
    const auto clock = [&](double change, bool& detected) {
        require(std::isfinite(change) && change >= 0,
            "Nonfinite mixed receiver field change.");
        const double noise = result.measured.relative_noise_floor;
        if (!(change > noise)) return std::numeric_limits<double>::infinity();
        detected = true;
        const double signal = std::sqrt(std::max(0., change*change-noise*noise));
        return signal > 0 ? dt/signal : std::numeric_limits<double>::infinity();
    };
    result.measured.timescale = std::min(
        clock(result.measured.potential_relative_change,
              result.measured.potential_signal_resolved),
        clock(result.measured.force_relative_change,
              result.measured.force_signal_resolved));
    require(result.measured.timescale > 0, "Invalid mixed receiver clock.");
    return result;
}
} // namespace venus_gpu_receiver
