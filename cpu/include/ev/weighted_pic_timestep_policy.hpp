#pragma once

// Select a global timestep from already evaluated radial-cell clocks.
// The optional budget counts deposited receiver mass whose clock may be
// excluded from the minimum. It changes the clock selection only: every
// particle, field source and random state remains in the physical update.
// Inputs include the reference deposited mass and the configured maximum.
// Outputs report the selected clock, controlling cell and complete excluded
// mass accounting, including the pre-existing activity-floor contribution.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ev { namespace weighted_pic_timestep_policy {

struct CellClock {
    std::size_t cell;
    double mass;
    double timestep;
};

struct Selection {
    double timestep = 0.0;
    long double reference_mass = 0.0L;
    long double activity_floor_mass = 0.0L;
    long double additional_omitted_mass = 0.0L;
    long double total_excluded_clock_mass = 0.0L;
    long double requested_budget_mass = 0.0L;
    long double total_excluded_fraction = 0.0L;
    std::size_t activity_floor_cells = 0;
    std::size_t additional_omitted_cells = 0;
    std::size_t controlling_cell = std::numeric_limits<std::size_t>::max();
    double shortest_unclipped_timestep = std::numeric_limits<double>::infinity();
    std::size_t shortest_unclipped_cell = std::numeric_limits<std::size_t>::max();
    bool maximum_timestep_controls = false;
    bool mass_budget_disabled = false;
    bool mass_diagnostics_available = false;
};

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// A positive budget includes mass already below the configured activity floor.
// Equal clocks are omitted as a group, keeping the decision independent of
// cell numbering. The optional budget applies to deposited mass, not objects.
inline Selection select_with_mass_budget(
    const std::vector<CellClock>& cells,
    double deposited_mass,
    double maximum_timestep,
    double unfiltered_timestep,
    double omitted_mass_fraction) {
    require(std::isfinite(omitted_mass_fraction)
        && omitted_mass_fraction >= 0.0 && omitted_mass_fraction < 1.0,
        "Require finite 0<=omitted mass fraction<1.");
    require(std::isfinite(maximum_timestep) && maximum_timestep > 0.0
        && std::isfinite(unfiltered_timestep) && unfiltered_timestep > 0.0
        && unfiltered_timestep <= maximum_timestep,
        "Invalid unfiltered timestep or maximum.");
    Selection result;
    result.timestep = unfiltered_timestep;
    // Production integration branches here before constructing the eligible clock list,
    // leaving the orbital zero-budget floating operation order literal.
    // A zero additional budget returns the supplied unfiltered interval
    // without rebuilding its arithmetic. The caller retains its default
    // clock path and floating-point evaluation order.
    if (omitted_mass_fraction == 0.0) {
        result.mass_budget_disabled = true;
        return result;
    }
    require(std::isfinite(deposited_mass) && deposited_mass > 0.0,
        "Positive budget requires finite positive deposited mass.");
    result.reference_mass = deposited_mass;
    result.mass_diagnostics_available = true;
    // Cells at or below 1e-12 of deposited mass already lie under the
    // activity floor. Their total mass consumes the same requested omission
    // budget rather than becoming an unreported extra exclusion.
    const double activity_floor = 1.0e-12 * deposited_mass;
    std::vector<CellClock> eligible;
    long double total = 0.0L;
    std::vector<std::size_t> ids;
    for (const CellClock& value : cells) {
        require(std::isfinite(value.mass) && value.mass >= 0.0
            && std::isfinite(value.timestep) && value.timestep > 0.0
            && value.timestep <= maximum_timestep,
            "Invalid cell mass or already evaluated clock.");
        ids.push_back(value.cell);
        total += static_cast<long double>(value.mass);
        if (!(value.mass > activity_floor)) {
            result.activity_floor_mass += static_cast<long double>(value.mass);
            ++result.activity_floor_cells;
        } else {
            eligible.push_back(value);
        }
    }
    std::sort(ids.begin(), ids.end());
    require(std::adjacent_find(ids.begin(), ids.end()) == ids.end(),
        "Cell identifiers must be unique.");
    const long double roundoff = 128.0L * std::numeric_limits<double>::epsilon()
        * std::max(1.0L, static_cast<long double>(cells.size()))
        * result.reference_mass;
    // Check that supplied cell masses close the deposited reference mass
    // before interpreting any omission fraction as a physical mass fraction.
    require(std::abs(total - result.reference_mass) <= roundoff,
        "Cell masses do not close the deposited reference mass.");
    result.requested_budget_mass =
        static_cast<long double>(omitted_mass_fraction) * result.reference_mass;
    require(result.activity_floor_mass <= result.requested_budget_mass,
        "Requested total budget is below pre-existing activity-floor mass.");
    // Visit clocks from shortest to longest. Cell ID is only a deterministic
    // tie order; the following loop treats every exactly equal clock as
    // one group so the physical decision is independent of numbering.
    std::sort(eligible.begin(), eligible.end(),
        [](const CellClock& first, const CellClock& second) {
            if (first.timestep != second.timestep)
                return first.timestep < second.timestep;
            return first.cell < second.cell;
        });
    result.timestep = maximum_timestep;
    result.total_excluded_clock_mass = result.activity_floor_mass;
    for (std::size_t first = 0; first < eligible.size();) {
        std::size_t end = first + 1;
        long double group_mass = eligible[first].mass;
        while (end < eligible.size()
            && eligible[end].timestep == eligible[first].timestep) {
            group_mass += static_cast<long double>(eligible[end].mass);
            ++end;
        }
        // A cap-limited group imposes no extra restriction and stays counted.
        // Exclude a complete shortest-clock group only when its additional mass
        // fits the remaining budget. The first group that cannot fit determines
        // the global step; particles in excluded groups are still evolved.
        if (eligible[first].timestep < maximum_timestep
            && result.total_excluded_clock_mass + group_mass
                <= result.requested_budget_mass) {
            result.additional_omitted_mass += group_mass;
            result.total_excluded_clock_mass += group_mass;
            result.additional_omitted_cells += end - first;
            first = end;
            continue;
        }
        result.timestep = eligible[first].timestep;
        result.controlling_cell = eligible[first].cell;
        break;
    }
    result.total_excluded_fraction =
        result.total_excluded_clock_mass / result.reference_mass;
    result.maximum_timestep_controls = result.timestep == maximum_timestep;
    if (result.maximum_timestep_controls)
        result.controlling_cell = std::numeric_limits<std::size_t>::max();
    require(result.total_excluded_clock_mass <= result.requested_budget_mass,
        "Cumulative excluded clock mass exceeds the requested budget.");
    require(result.timestep >= unfiltered_timestep,
        "Mass selection unexpectedly shortened the unfiltered timestep.");
    return result;
}

// Report prepared gravitational clocks when the additional mass budget is off.
// The orbital zero-budget path returns before constructing these clocks.
// Describe a previously evaluated gravitational minimum without changing
// it. This reports its controller and activity-floor mass when the
// additional mass-budget selection is disabled.
inline Selection describe_unfiltered_clocks(
    const std::vector<CellClock>& cells,
    double deposited_mass,
    double maximum_timestep,
    double unfiltered_timestep) {
    require(std::isfinite(deposited_mass) && deposited_mass > 0.0
        && std::isfinite(maximum_timestep) && maximum_timestep > 0.0
        && std::isfinite(unfiltered_timestep) && unfiltered_timestep > 0.0
        && unfiltered_timestep <= maximum_timestep,
        "Invalid unfiltered clock diagnostics.");
    Selection result;
    result.timestep = unfiltered_timestep;
    result.reference_mass = deposited_mass;
    result.mass_budget_disabled = true;
    result.mass_diagnostics_available = true;
    const double floor = 1e-12 * deposited_mass;
    for (const CellClock& value : cells) {
        require(std::isfinite(value.mass) && value.mass >= 0.0
            && std::isfinite(value.timestep) && value.timestep > 0.0,
            "Invalid prepared clock diagnostics.");
        if (!(value.mass > floor)) {
            result.activity_floor_mass += static_cast<long double>(value.mass);
            ++result.activity_floor_cells;
        } else if (value.timestep == unfiltered_timestep
            && result.controlling_cell == std::numeric_limits<std::size_t>::max()) {
            result.controlling_cell = value.cell;
        }
    }
    result.total_excluded_clock_mass = result.activity_floor_mass;
    result.total_excluded_fraction = result.activity_floor_mass / result.reference_mass;
    result.maximum_timestep_controls = result.timestep == maximum_timestep;
    if (result.maximum_timestep_controls)
        result.controlling_cell = std::numeric_limits<std::size_t>::max();
    return result;
}

// Separate optional policy: exact free angular drift removes the coordinate
// centrifugal stiffness; gravity and expansion still bound the global step.
// Collision exposure and field-change checks constrain the coupled update separately.
// For exact angular drift, the caller supplies gravity and background
// reference frequencies separately. Their squared sum gives the remaining
// force/background timescale; the caller owns its physical domain and
// any conservative background reference used for static data.
inline double force_h_clock(
    double gravity_frequency_squared,
    double hubble_squared,
    double dynamical_fraction,
    double maximum_timestep,
    bool exact_angular_drift) {
    require(exact_angular_drift, "Force/H policy requires exact angular drift.");
    require(std::isfinite(gravity_frequency_squared)
        && gravity_frequency_squared >= 0.0
        && std::isfinite(hubble_squared) && hubble_squared >= 0.0
        && std::isfinite(dynamical_fraction) && dynamical_fraction > 0.0
        && std::isfinite(maximum_timestep) && maximum_timestep > 0.0,
        "Invalid force/H clock input.");
    const double frequency_squared = gravity_frequency_squared + hubble_squared;
    require(std::isfinite(frequency_squared), "Force/H frequency overflow.");
    return frequency_squared > 0.0
        ? std::min(maximum_timestep,
            dynamical_fraction / std::sqrt(frequency_squared))
        : maximum_timestep;
}

} } // namespace ev::weighted_pic_timestep_policy
