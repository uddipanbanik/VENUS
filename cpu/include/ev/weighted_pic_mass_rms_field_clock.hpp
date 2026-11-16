#pragma once

// Measure field variation with a receiver-mass-weighted RMS norm.
// Inputs are two reconstructed potential/force arrays, shell receiver masses,
// effective marker counts and their observation interval. Outputs are relative
// field changes, a sampling detection floor and a resolved variation time.
// Every receiver mass participates in the norm. The calling driver separately
// checks whether this mesh representation covers all physical receivers.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ev { namespace weighted_pic_field_clock {

struct Measurement {
    double timescale = std::numeric_limits<double>::infinity();
    double potential_relative_change = 0.0;
    double force_relative_change = 0.0;
    double relative_noise_floor = 0.0;
    double effective_marker_number = 0.0;
    long double receiver_mass = 0.0L;
    bool potential_signal_resolved = false;
    bool force_signal_resolved = false;
};

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// All receiver masses contribute. Fields include the complete Poisson source.
// The noise coefficient and effective-number definition equal the maximum-norm
// policy. Each channel is detected relative to this sampling floor.
inline Measurement mass_rms_change_clock(
    const std::vector<double>& previous_potential,
    const std::vector<double>& previous_force,
    const std::vector<double>& current_potential,
    const std::vector<double>& current_force,
    const std::vector<double>& shell_mass,
    const std::vector<double>& effective_marker_number,
    double observation_interval) {
    const std::size_t cells = shell_mass.size();
    require(std::isfinite(observation_interval) && observation_interval > 0.0
        && previous_potential.size() == cells
        && previous_force.size() == cells
        && current_potential.size() == cells
        && current_force.size() == cells
        && effective_marker_number.size() == cells,
        "Invalid receiver-mass field-clock dimensions or interval.");
    Measurement result;
    // The global effective count is M^2/sum(M_i^2/N_eff_i), where M is
    // total receiver mass. This combines unequal shell sampling weights
    // without identifying numerical marker count with physical particle count.
    long double squared_marker_mass = 0.0L;
    for (std::size_t i = 0; i < cells; ++i) {
        require(std::isfinite(shell_mass[i]) && shell_mass[i] >= 0.0
            && std::isfinite(effective_marker_number[i])
            && effective_marker_number[i] >= 0.0
            && std::isfinite(previous_potential[i])
            && std::isfinite(previous_force[i])
            && std::isfinite(current_potential[i])
            && std::isfinite(current_force[i]),
            "Nonfinite or negative receiver-mass field-clock input.");
        if (shell_mass[i] > 0.0) {
            require(effective_marker_number[i] > 0.0,
                "Positive receiver mass requires effective marker support.");
            result.receiver_mass += shell_mass[i];
            squared_marker_mass += static_cast<long double>(shell_mass[i])
                * shell_mass[i] / effective_marker_number[i];
        }
    }
    if (!(result.receiver_mass > 0.0L)) return result;
    require(squared_marker_mass > 0.0L,
        "Receiver-mass field-clock sampling measure vanished.");
    result.effective_marker_number = static_cast<double>(
        result.receiver_mass * result.receiver_mass / squared_marker_mass);
    require(std::isfinite(result.effective_marker_number)
        && result.effective_marker_number > 0.0,
        "Invalid field-clock effective marker number.");
    // A quarter of N_eff^(-1/2) supplies the configured sampling detection
    // floor; 4096 machine epsilons supplies its roundoff floor. Changes below
    // this level do not define a measured finite-difference derivative.
    result.relative_noise_floor = std::max(
        4096.0 * std::numeric_limits<double>::epsilon(),
        0.25 / std::sqrt(result.effective_marker_number));
    long double previous_phi2 = 0.0L, current_phi2 = 0.0L, difference_phi2 = 0.0L;
    long double previous_force2 = 0.0L, current_force2 = 0.0L, difference_force2 = 0.0L;
    for (std::size_t i = 0; i < cells; ++i) {
        // Normalize positive receiver masses to weights summing to one.
        // Measure both field norms and their difference with the same weights,
        // so a sparsely represented radial cell is weighted by its receiving mass.
        const long double weight = shell_mass[i] / result.receiver_mass;
        const long double p0 = previous_potential[i], p1 = current_potential[i];
        const long double g0 = previous_force[i], g1 = current_force[i];
        previous_phi2 += weight * p0 * p0;
        current_phi2 += weight * p1 * p1;
        difference_phi2 += weight * (p1 - p0) * (p1 - p0);
        previous_force2 += weight * g0 * g0;
        current_force2 += weight * g1 * g1;
        difference_force2 += weight * (g1 - g0) * (g1 - g0);
    }
    // Normalize the difference by the larger before/after field norm.
    // A channel that vanishes at both endpoints has zero relative change.
    auto relative = [](long double before, long double after, long double difference) {
        const long double norm2 = std::max(before, after);
        return norm2 > 0.0L ? static_cast<double>(std::sqrt(difference / norm2)) : 0.0;
    };
    result.potential_relative_change = relative(previous_phi2, current_phi2, difference_phi2);
    result.force_relative_change = relative(previous_force2, current_force2, difference_force2);
    auto resolved_clock = [&](double change, bool& resolved) {
        require(std::isfinite(change) && change >= 0.0,
            "Invalid mass-RMS relative field change.");
        if (!(change > result.relative_noise_floor))
            return std::numeric_limits<double>::infinity();
        resolved = true;
        // Subtract the sampling floor in quadrature only after a channel
        // exceeds it. The observation interval divided by this resolved relative
        // change is the field-variation time used by the caller.
        const double signal = std::sqrt(std::max(0.0,
            change * change - result.relative_noise_floor * result.relative_noise_floor));
        return signal > 0.0 ? observation_interval / signal
            : std::numeric_limits<double>::infinity();
    };
    // Potential and force are separate measured channels. The shorter
    // resolved variation time constrains the next field/collision interval;
    // positive infinity represents two unresolved or stationary channels.
    result.timescale = std::min(
        resolved_clock(result.potential_relative_change, result.potential_signal_resolved),
        resolved_clock(result.force_relative_change, result.force_signal_resolved));
    require((std::isfinite(result.timescale) && result.timescale > 0.0)
        || (std::isinf(result.timescale) && result.timescale > 0.0),
        "Mass-RMS field clock selected an invalid time.");
    return result;
}

} } // namespace ev::weighted_pic_field_clock
