#pragma once

// Describe radial travel by a periodic phase in a frozen spherical potential.
// Inputs are specific energy H, eta=L^2/2, an effective-potential evaluator,
// radial search points, and any unresolved central return delay. Outputs are
// turning radii, period, phase/location conversions, and periodic row integrals.
// The phase s is elapsed orbital time divided by the full radial period T,
// so collisionless motion advances s by dt/T at fixed H and eta. Turning-point
// quadrature uses r=r_peri+(r_apo-r_peri)*sin(alpha)^2 to keep travel-time
// integrals regular. Inner-crossing orbits append their unresolved return time
// after the two visible radial legs, preserving the full physical period.

#include "ev/phase_space_cells.hpp"
#include "ev/radial_orbit_regularization.hpp"

#include <numeric>

namespace ev {

// Cache one energy/angular-momentum orbit. pericenter denotes the lower
// represented radius for an inner-crossing orbit, whose true central segment
// is represented separately by hidden_return_time.
struct HamiltonianOrbitPhaseGeometry {
    double eta = 0.0;
    double energy = 0.0;
    double pericenter = 0.0;
    double apocenter = 0.0;
    bool inner_is_turning_point = true;
    double visible_outgoing_time = 0.0;
    double hidden_return_time = 0.0;
    double radial_period = 0.0;
    std::vector<double> alpha_edge;
    std::vector<double> outgoing_time_edge;
    double time_quadrature_scale = 1.0;
    // A visible location has physical (r,w). A hidden location instead
    // carries the remaining delay before its return to the resolved boundary.
    struct Location {
        CharacteristicState state;
        bool visible = true;
        double hidden_remaining_time = 0.0;
    };
    std::vector<Location> phase_center;
};

// Store nonnegative cell averages on a uniform periodic s grid. A single
// offset moves the row through any number of periods without remapping it.
struct HamiltonianOrbitPhaseRow {
    double eta = 0.0;
    double energy = 0.0;
    double radial_period = 0.0;
    std::vector<double> cell_average;
    // f(s,t)=f_0(s-phase_offset), with s in [0,1).
    double phase_offset = 0.0;
};

struct HamiltonianOrbitExactPhase {
    double phase = 0.0;
    double radial_period = 0.0;
    double pericenter = 0.0;
    double apocenter = 0.0;
};

// Reduce finite phase to [0,1), retaining periodicity for either sign of
// the requested interval and for advances longer than one full orbit.
inline double hamiltonian_orbit_wrap_phase(double phase) {
    if (!std::isfinite(phase)) {
        throw std::runtime_error("Orbit phase must be finite.");
    }
    double wrapped = phase - std::floor(phase);
    if (wrapped >= 1.0) {
        wrapped = 0.0;
    }
    return wrapped;
}

// Require a physical eta, finite energy, positive finite period, and a
// nonnegative represented row before shifting or integrating its phase.
inline void validate_hamiltonian_orbit_phase_row(
    const HamiltonianOrbitPhaseRow& row) {
    if (!(row.eta >= 0.0) || !std::isfinite(row.eta)
        || !std::isfinite(row.energy)
        || !(row.radial_period > 0.0) || !std::isfinite(row.radial_period)
        || row.cell_average.empty()
        || !(row.phase_offset >= 0.0) || !(row.phase_offset < 1.0)
        || !std::isfinite(row.phase_offset)) {
        throw std::runtime_error("Invalid Hamiltonian orbit-phase row.");
    }
    for (double value : row.cell_average) {
        if (!(value >= 0.0) || !std::isfinite(value)) {
            throw std::runtime_error(
                "Hamiltonian orbit-phase row contains an invalid cell average.");
        }
    }
}

// Frozen-orbit advection changes only the phase offset: ds/dt=1/T.
// The stored cell averages and their mass remain fixed during this shift.
inline void advance_hamiltonian_orbit_phase_row(
    HamiltonianOrbitPhaseRow& row,
    double dt) {
    validate_hamiltonian_orbit_phase_row(row);
    if (!std::isfinite(dt)) {
        throw std::runtime_error("Orbit-phase interval must be finite.");
    }
    row.phase_offset = hamiltonian_orbit_wrap_phase(
        row.phase_offset + dt / row.radial_period);
}

// Equal phase-cell widths make the arithmetic row mean its full-period
// integral. Accumulate in extended precision before returning binary64.
inline double hamiltonian_orbit_phase_row_mean(
    const HamiltonianOrbitPhaseRow& row) {
    validate_hamiltonian_orbit_phase_row(row);
    const long double sum = std::accumulate(
        row.cell_average.begin(), row.cell_average.end(), 0.0L);
    return static_cast<double>(
        sum / static_cast<long double>(row.cell_average.size()));
}

// Integrate the periodic piecewise-constant row over an arbitrary interval.
// Full periods use the conserved mean; the remaining segment crosses each
// intersected phase cell with its exact overlap width.
inline double hamiltonian_orbit_phase_integral_unshifted(
    const HamiltonianOrbitPhaseRow& row,
    double phase_begin,
    double phase_end) {
    validate_hamiltonian_orbit_phase_row(row);
    if (!std::isfinite(phase_begin) || !std::isfinite(phase_end)
        || phase_end < phase_begin) {
        throw std::runtime_error("Invalid orbit-phase integration interval.");
    }
    const double length = phase_end - phase_begin;
    if (length == 0.0) {
        return 0.0;
    }
    const double periods = std::floor(length);
    long double integral = static_cast<long double>(periods)
        * hamiltonian_orbit_phase_row_mean(row);
    double remaining = length - periods;
    double phase = hamiltonian_orbit_wrap_phase(phase_begin);
    const double cells = static_cast<double>(row.cell_average.size());
    while (remaining > 0.0) {
        int cell = static_cast<int>(std::floor(phase * cells));
        cell = std::clamp(
            cell, 0, static_cast<int>(row.cell_average.size()) - 1);
        const double right = static_cast<double>(cell + 1) / cells;
        double width = std::min(remaining, right - phase);
        if (!(width > 0.0)) {
            phase = 0.0;
            continue;
        }
        integral += static_cast<long double>(width)
            * row.cell_average[static_cast<std::size_t>(cell)];
        remaining -= width;
        phase += width;
        if (phase >= 1.0
            - 8.0 * std::numeric_limits<double>::epsilon()) {
            phase = 0.0;
        }
    }
    return static_cast<double>(integral);
}

// Translate target phase limits into the stored row coordinate using its
// accumulated offset before evaluating the periodic overlap integral.
inline double hamiltonian_orbit_phase_integral(
    const HamiltonianOrbitPhaseRow& row,
    double phase_begin,
    double phase_end) {
    return hamiltonian_orbit_phase_integral_unshifted(
        row,
        phase_begin - row.phase_offset,
        phase_end - row.phase_offset);
}

template <class EffectivePotential>
// A radial turning point satisfies Phi_eff(r)=H, where
// Phi_eff=Phi+eta/r^2. A sign-changing bracket gives bounded bisection.
inline double hamiltonian_orbit_bisect_turning_point(
    double left,
    double right,
    double energy,
    EffectivePotential&& effective_potential) {
    double f_left = effective_potential(left) - energy;
    double f_right = effective_potential(right) - energy;
    if (f_left == 0.0) {
        return left;
    }
    if (f_right == 0.0) {
        return right;
    }
    if ((f_left < 0.0) == (f_right < 0.0)) {
        throw std::runtime_error("Turning-point interval does not bracket a root.");
    }
    for (int iteration = 0; iteration < 80; ++iteration) {
        const double middle = 0.5 * (left + right);
        const double f_middle = effective_potential(middle) - energy;
        if (f_middle == 0.0) {
            return middle;
        }
        if ((f_middle < 0.0) == (f_left < 0.0)) {
            left = middle;
            f_left = f_middle;
        } else {
            right = middle;
            f_right = f_middle;
        }
        const double scale = std::max({1.0, std::abs(left), std::abs(right)});
        if (right - left
            <= 64.0 * std::numeric_limits<double>::epsilon() * scale) {
            break;
        }
    }
    return 0.5 * (left + right);
}

template <class EffectivePotential>
// Scan ordered positive radii for effective-potential crossings, refine
// them, and merge duplicate roots introduced by shared scan endpoints.
inline std::array<double, 2> hamiltonian_orbit_find_turning_points(
    const std::vector<double>& scan_radius,
    double energy,
    EffectivePotential&& effective_potential) {
    if (scan_radius.size() < 3 || !std::isfinite(energy)) {
        throw std::runtime_error("Orbit turning-point scan is incomplete.");
    }
    std::vector<double> root;
    for (std::size_t q = 0; q + 1 < scan_radius.size(); ++q) {
        const double left = scan_radius[q];
        const double right = scan_radius[q + 1];
        if (!(left > 0.0) || !(right > left)) {
            throw std::runtime_error(
                "Orbit turning-point radii must be strictly increasing and positive.");
        }
        const double f_left = effective_potential(left) - energy;
        const double f_right = effective_potential(right) - energy;
        if (!std::isfinite(f_left) || !std::isfinite(f_right)) {
            throw std::runtime_error(
                "Orbit turning-point scan encountered a non-finite potential.");
        }
        if (f_left == 0.0) {
            root.push_back(left);
        }
        if ((f_left < 0.0) != (f_right < 0.0)) {
            root.push_back(hamiltonian_orbit_bisect_turning_point(
                left, right, energy, effective_potential));
        }
        if (q + 2 == scan_radius.size() && f_right == 0.0) {
            root.push_back(right);
        }
    }
    std::sort(root.begin(), root.end());
    root.erase(
        std::unique(
            root.begin(),
            root.end(),
            [](double first, double second) {
                return std::abs(first - second)
                    <= 128.0 * std::numeric_limits<double>::epsilon()
                        * std::max({1.0, std::abs(first), std::abs(second)});
            }),
        root.end());
    if (root.size() < 2) {
        throw std::runtime_error(
            "The represented domain does not contain both radial turning points.");
    }
    return {root.front(), root.back()};
}

template <class EffectivePotential>
// Compute integral dr/sqrt(2*(H-Phi_eff)) over one outgoing segment.
// The sin^2 radial map removes the square-root endpoint singularities;
// a lower resolved face can instead be an ordinary interior point of the orbit.
inline double hamiltonian_orbit_time_integral_alpha(
    double alpha_begin,
    double alpha_end,
    int panels,
    double pericenter,
    double apocenter,
    double energy,
    EffectivePotential&& effective_potential,
    bool inner_is_turning_point = true) {
    if (!(alpha_end >= alpha_begin) || panels < 1) {
        throw std::runtime_error("Invalid orbit-time quadrature interval.");
    }
    if (alpha_end == alpha_begin) {
        return 0.0;
    }
    const auto& rule = phase_space_gauss_legendre_rule(12);
    const double delta_r = apocenter - pericenter;
    // Turning radii are rounded roots. Remove their small endpoint energy
    // defects linearly so the regularized quadrature retains finite limits.
    const double pericenter_defect = inner_is_turning_point
        ? effective_potential(pericenter) - energy
        : 0.0;
    const double apocenter_defect =
        effective_potential(apocenter) - energy;
    const double energy_scale = std::max(1.0, std::abs(energy));
    const double resolvable_energy = 256.0
        * std::numeric_limits<double>::epsilon() * energy_scale;
    const double middle_available = energy
        - effective_potential(0.5 * (pericenter + apocenter))
        + 0.5 * (pericenter_defect + apocenter_defect);
    auto available_at_fraction = [&](double fraction) {
        const double radius = std::fma(delta_r, fraction, pericenter);
        const double endpoint_defect =
            (1.0 - fraction) * pericenter_defect
            + fraction * apocenter_defect;
        return energy - effective_potential(radius) + endpoint_defect;
    };
    // Factor available energy by (r-r_peri)*(r_apo-r), leaving a regular
    // positive function whose square root determines the alpha-time Jacobian.
    auto turning_factor = [&](double fraction) {
        const double denominator = delta_r * delta_r
            * fraction * (1.0 - fraction);
        const double available = available_at_fraction(fraction);
        if (available > resolvable_energy && denominator > 0.0) {
            return available / denominator;
        }

        // E-Phi_eff loses all significant digits within a few ulps of a
        // turning point.  Q=(E-Phi_eff)/[(r-rp)(ra-r)] remains regular there.
        // Extrapolate Q from two nearby, fully resolvable interior points.
        if (!(middle_available > 0.0) || !(denominator > 0.0)) {
            throw std::runtime_error(
                "Orbit turning-point factor is not numerically resolvable.");
        }
        const double probe = std::clamp(
            8.0 * resolvable_energy / middle_available,
            1.0e-6,
            0.2);
        const bool near_left = fraction <= 0.5;
        const double first_fraction = near_left ? probe : 1.0 - probe;
        const double second_probe = std::min(2.0 * probe, 0.4);
        const double second_fraction = near_left
            ? second_probe : 1.0 - second_probe;
        auto factor_at = [&](double value) {
            const double value_denominator = delta_r * delta_r
                * value * (1.0 - value);
            const double value_available = available_at_fraction(value);
            if (!(value_available > 0.0)
                || !(value_denominator > 0.0)) {
                throw std::runtime_error(
                    "Orbit turning-point factor probe is invalid.");
            }
            return value_available / value_denominator;
        };
        const double first_factor = factor_at(first_fraction);
        const double second_factor = factor_at(second_fraction);
        const double slope = (second_factor - first_factor)
            / (second_fraction - first_fraction);
        const double extrapolated = first_factor
            + slope * (fraction - first_fraction);
        if (!(extrapolated > 0.0) || !std::isfinite(extrapolated)) {
            throw std::runtime_error(
                "Orbit turning-point factor extrapolation is invalid.");
        }
        return extrapolated;
    };
    long double integral = 0.0L;
    for (int panel = 0; panel < panels; ++panel) {
        const double left = alpha_begin
            + (alpha_end - alpha_begin)
                * static_cast<double>(panel) / static_cast<double>(panels);
        const double right = alpha_begin
            + (alpha_end - alpha_begin)
                * static_cast<double>(panel + 1) / static_cast<double>(panels);
        const double center = 0.5 * (left + right);
        const double half_width = 0.5 * (right - left);
        for (std::size_t q = 0; q < rule.first.size(); ++q) {
            const double alpha = center + half_width * rule.first[q];
            const double sine = std::sin(alpha);
            const double cosine = std::cos(alpha);
            const double radial_fraction = sine * sine;
            const double radius = std::fma(
                delta_r, radial_fraction, pericenter);
            const double dr_dalpha = 2.0 * delta_r * sine * cosine;
            const double endpoint_defect =
                (1.0 - radial_fraction) * pericenter_defect
                + radial_fraction * apocenter_defect;
            // With two true turning endpoints the radial-map derivative
            // cancels their singular factors analytically. A crossing face
            // uses the ordinary dr/dalpha divided by radial speed below.
            if (inner_is_turning_point) {
                const double factor = turning_factor(radial_fraction);
                integral += static_cast<long double>(
                    half_width * rule.second[q] * std::sqrt(2.0 / factor));
                continue;
            }
            double radial_speed2 = 2.0
                * (energy - effective_potential(radius) + endpoint_defect);
            if (radial_speed2 < -1024.0
                    * std::numeric_limits<double>::epsilon() * energy_scale) {
                throw std::runtime_error(
                    "Orbit-time quadrature left the allowed radial interval.");
            }
            radial_speed2 = std::max(
                radial_speed2, std::numeric_limits<double>::min());
            integral += static_cast<long double>(half_width * rule.second[q])
                * dr_dalpha / std::sqrt(radial_speed2);
        }
    }
    return static_cast<double>(integral);
}

/// Computes only the radial period.  Energy-row placement and stationary
/// H-integrals do not need the much larger phase-location table built by
/// build_hamiltonian_orbit_phase_geometry().
template <class EffectivePotential>
inline double hamiltonian_orbit_radial_period(
    double energy,
    const std::vector<double>& scan_radius,
    EffectivePotential&& effective_potential,
    double hidden_return_time =
        std::numeric_limits<double>::quiet_NaN(),
    int quadrature_panels = 64) {
    if (!std::isfinite(energy) || scan_radius.size() < 3
        || quadrature_panels < 1) {
        throw std::runtime_error("Invalid Hamiltonian orbit-period request.");
    }
    bool inner_is_turning_point = true;
    std::array<double, 2> turning{};
    const double inner_value = effective_potential(scan_radius.front()) - energy;
    if (inner_value < 0.0 && std::isfinite(hidden_return_time)
        && hidden_return_time >= 0.0) {
        inner_is_turning_point = false;
        turning[0] = scan_radius.front();
        bool found_outer = false;
        for (std::size_t q = 0; q + 1 < scan_radius.size(); ++q) {
            const double left_value =
                effective_potential(scan_radius[q]) - energy;
            const double right_value =
                effective_potential(scan_radius[q + 1]) - energy;
            if (left_value <= 0.0 && right_value >= 0.0) {
                turning[1] = hamiltonian_orbit_bisect_turning_point(
                    scan_radius[q],
                    scan_radius[q + 1],
                    energy,
                    effective_potential);
                found_outer = true;
                break;
            }
        }
        if (!found_outer) {
            throw std::runtime_error(
                "Inner-crossing orbit has no represented outer turning point.");
        }
    } else {
        turning = hamiltonian_orbit_find_turning_points(
            scan_radius, energy, effective_potential);
        hidden_return_time = 0.0;
    }
    const double half_period = hamiltonian_orbit_time_integral_alpha(
        0.0,
        0.5 * PI,
        quadrature_panels,
        turning[0],
        turning[1],
        energy,
        effective_potential,
        inner_is_turning_point);
    // Two visible legs plus the central delay form one complete return
    // to the same outward boundary/turning state.
    const double radial_period = 2.0 * half_period + hidden_return_time;
    if (!(radial_period > 0.0) || !std::isfinite(radial_period)) {
        throw std::runtime_error("Orbit radial period is invalid.");
    }
    return radial_period;
}

/// Computes the phase of one state without constructing a reusable orbit
/// table.  Collision products need one such value only once.  The same
/// r=rp+(ra-rp)sin^2(alpha) regularization used by the table builder removes
/// both turning-point square-root singularities, while this path evaluates
/// only the complete half-period and the requested partial travel time.
template <class EffectivePotential>
inline HamiltonianOrbitExactPhase hamiltonian_orbit_exact_phase_of_state(
    double energy,
    const CharacteristicState& state,
    const std::vector<double>& scan_radius,
    EffectivePotential&& effective_potential,
    double hidden_return_time =
        std::numeric_limits<double>::quiet_NaN(),
    int quadrature_panels = 48) {
    if (!std::isfinite(energy) || !(state.r > 0.0)
        || !std::isfinite(state.r) || !std::isfinite(state.w)
        || scan_radius.size() < 3 || quadrature_panels < 8) {
        throw std::runtime_error("Invalid exact orbit-phase request.");
    }
    bool inner_is_turning_point = true;
    std::array<double, 2> turning{};
    const double inner_value = effective_potential(scan_radius.front()) - energy;
    if (inner_value < 0.0 && std::isfinite(hidden_return_time)
        && hidden_return_time >= 0.0) {
        inner_is_turning_point = false;
        turning[0] = scan_radius.front();
        bool found_outer = false;
        for (std::size_t q = 0; q + 1 < scan_radius.size(); ++q) {
            const double left_value =
                effective_potential(scan_radius[q]) - energy;
            const double right_value =
                effective_potential(scan_radius[q + 1]) - energy;
            if (left_value <= 0.0 && right_value >= 0.0) {
                turning[1] = hamiltonian_orbit_bisect_turning_point(
                    scan_radius[q], scan_radius[q + 1], energy,
                    effective_potential);
                found_outer = true;
                break;
            }
        }
        if (!found_outer) {
            throw std::runtime_error(
                "Inner-crossing exact phase has no outer turning point.");
        }
    } else {
        turning = hamiltonian_orbit_find_turning_points(
            scan_radius, energy, effective_potential);
        hidden_return_time = 0.0;
    }
    const double radial_span = turning[1] - turning[0];
    const double radius_scale = std::max({1.0, std::abs(turning[0]),
                                          std::abs(turning[1])});
    const double radius_tolerance = 512.0
        * std::numeric_limits<double>::epsilon() * radius_scale;
    if (state.r < turning[0] - radius_tolerance
        || state.r > turning[1] + radius_tolerance) {
        throw std::runtime_error(
            "Exact orbit-phase state lies outside its turning points.");
    }
    const double radial_fraction = std::clamp(
        (state.r - turning[0]) / radial_span, 0.0, 1.0);
    const double alpha = std::asin(std::sqrt(radial_fraction));
    const double half_period = hamiltonian_orbit_time_integral_alpha(
        0.0, 0.5 * PI, quadrature_panels,
        turning[0], turning[1], energy, effective_potential,
        inner_is_turning_point);
    const int partial_panels = std::max(
        8,
        static_cast<int>(std::ceil(
            quadrature_panels * alpha / (0.5 * PI))));
    const double outgoing_time = alpha > 0.0
        ? hamiltonian_orbit_time_integral_alpha(
            0.0, alpha, partial_panels,
            turning[0], turning[1], energy, effective_potential,
            inner_is_turning_point)
        : 0.0;
    // Two visible legs plus the central delay form one complete return
    // to the same outward boundary/turning state.
    const double radial_period = 2.0 * half_period + hidden_return_time;
    if (!(half_period > 0.0) || !(radial_period > 0.0)
        || !std::isfinite(outgoing_time)
        || !std::isfinite(radial_period)) {
        throw std::runtime_error("Exact orbit phase is not finite.");
    }
    // Velocity sign distinguishes outward and inward visits to the same
    // radius. Inward travel follows the apocenter before reaching the center.
    const double physical_time = state.w < 0.0
        ? 2.0 * half_period - outgoing_time
        : outgoing_time;
    return {
        hamiltonian_orbit_wrap_phase(physical_time / radial_period),
        radial_period,
        turning[0],
        turning[1],
    };
}

template <class EffectivePotential>
// Reuse the cumulative time at the left table knot and integrate only the
// final alpha subinterval, keeping this observation consistent with the table.
inline double hamiltonian_orbit_outgoing_time_at_alpha(
    const HamiltonianOrbitPhaseGeometry& geometry,
    double alpha,
    EffectivePotential&& effective_potential) {
    if (geometry.alpha_edge.size() < 2
        || geometry.alpha_edge.size() != geometry.outgoing_time_edge.size()
        || !(alpha >= 0.0) || !(alpha <= 0.5 * PI)) {
        throw std::runtime_error("Invalid cached orbit-time table.");
    }
    const auto upper = std::upper_bound(
        geometry.alpha_edge.begin(), geometry.alpha_edge.end(), alpha);
    int panel = static_cast<int>(upper - geometry.alpha_edge.begin()) - 1;
    panel = std::clamp(
        panel, 0, static_cast<int>(geometry.alpha_edge.size()) - 2);
    return geometry.outgoing_time_edge[static_cast<std::size_t>(panel)]
        + geometry.time_quadrature_scale
            * hamiltonian_orbit_time_integral_alpha(
                geometry.alpha_edge[static_cast<std::size_t>(panel)],
                alpha,
                1,
                geometry.pericenter,
                geometry.apocenter,
                geometry.energy,
                effective_potential,
                geometry.inner_is_turning_point);
}

template <class EffectivePotential>
inline HamiltonianOrbitPhaseGeometry::Location
// Invert visible travel time by bracketed bisection inside its cached
// alpha panel. Hidden phases return a delay record for the central carrier.
hamiltonian_orbit_location_at_phase(
    const HamiltonianOrbitPhaseGeometry& geometry,
    double phase,
    EffectivePotential&& effective_potential) {
    phase = hamiltonian_orbit_wrap_phase(phase);
    const double physical_time = phase * geometry.radial_period;
    // The final part of a full phase cycle belongs to the unresolved
    // central excursion, rather than a fictitious resolved-grid position.
    if (physical_time > 2.0 * geometry.visible_outgoing_time) {
        return {
            {geometry.pericenter, 0.0},
            false,
            geometry.radial_period - physical_time,
        };
    }
    const bool outgoing = physical_time <= geometry.visible_outgoing_time;
    const double target_time = outgoing
        ? physical_time
        : 2.0 * geometry.visible_outgoing_time - physical_time;
    const auto upper = std::lower_bound(
        geometry.outgoing_time_edge.begin(),
        geometry.outgoing_time_edge.end(),
        target_time);
    int panel = static_cast<int>(upper - geometry.outgoing_time_edge.begin()) - 1;
    panel = std::clamp(
        panel, 0, static_cast<int>(geometry.alpha_edge.size()) - 2);
    double lower_alpha = geometry.alpha_edge[static_cast<std::size_t>(panel)];
    double upper_alpha = geometry.alpha_edge[static_cast<std::size_t>(panel + 1)];
    for (int iteration = 0; iteration < 52; ++iteration) {
        const double middle = 0.5 * (lower_alpha + upper_alpha);
        if (hamiltonian_orbit_outgoing_time_at_alpha(
                geometry, middle, effective_potential) < target_time) {
            lower_alpha = middle;
        } else {
            upper_alpha = middle;
        }
    }
    const double alpha = 0.5 * (lower_alpha + upper_alpha);
    const double sine = std::sin(alpha);
    const double radius = geometry.pericenter
        + (geometry.apocenter - geometry.pericenter) * sine * sine;
    const double speed = std::sqrt(std::max(
        0.0,
        2.0 * (geometry.energy - effective_potential(radius))));
    return {{radius, outgoing ? speed : -speed}, true, 0.0};
}

/// Inverts the cached monotone orbit-time table with safeguarded secant
/// iterations.  Atlas construction calls this routine at many shared phase
/// vertices, so reusing the table avoids dozens of fresh orbit integrals per
/// vertex while computing velocity from the same Hamiltonian energy relation.
template <class EffectivePotential>
inline HamiltonianOrbitPhaseGeometry::Location
hamiltonian_orbit_location_at_phase_cached(
    const HamiltonianOrbitPhaseGeometry& geometry,
    double phase,
    EffectivePotential&& effective_potential) {
    phase = hamiltonian_orbit_wrap_phase(phase);
    const double physical_time = phase * geometry.radial_period;
    // The final part of a full phase cycle belongs to the unresolved
    // central excursion, rather than a fictitious resolved-grid position.
    if (physical_time > 2.0 * geometry.visible_outgoing_time) {
        return {
            {geometry.pericenter, 0.0},
            false,
            geometry.radial_period - physical_time,
        };
    }
    const bool outgoing = physical_time <= geometry.visible_outgoing_time;
    const double target_time = outgoing
        ? physical_time
        : 2.0 * geometry.visible_outgoing_time - physical_time;
    if (target_time <= 0.0) {
        const double boundary_speed = std::sqrt(std::max(
            0.0,
            2.0 * (
                geometry.energy
                - effective_potential(geometry.pericenter))));
        return {{
            geometry.pericenter,
            outgoing ? boundary_speed : -boundary_speed,
        }, true, 0.0};
    }
    if (target_time >= geometry.visible_outgoing_time) {
        return {{geometry.apocenter, 0.0}, true, 0.0};
    }
    const auto upper = std::lower_bound(
        geometry.outgoing_time_edge.begin(),
        geometry.outgoing_time_edge.end(),
        target_time);
    int panel = static_cast<int>(upper - geometry.outgoing_time_edge.begin()) - 1;
    panel = std::clamp(
        panel, 0, static_cast<int>(geometry.alpha_edge.size()) - 2);
    double lower_alpha = geometry.alpha_edge[static_cast<std::size_t>(panel)];
    double upper_alpha = geometry.alpha_edge[static_cast<std::size_t>(panel + 1)];
    double lower_time =
        geometry.outgoing_time_edge[static_cast<std::size_t>(panel)];
    double upper_time =
        geometry.outgoing_time_edge[static_cast<std::size_t>(panel + 1)];
    double alpha = lower_alpha
        + (upper_alpha - lower_alpha)
            * (target_time - lower_time) / (upper_time - lower_time);
    for (int iteration = 0; iteration < 14; ++iteration) {
        const double represented_time = hamiltonian_orbit_outgoing_time_at_alpha(
            geometry, alpha, effective_potential);
        if (represented_time < target_time) {
            lower_alpha = alpha;
            lower_time = represented_time;
        } else {
            upper_alpha = alpha;
            upper_time = represented_time;
        }
        const double denominator = upper_time - lower_time;
        double next_alpha = denominator > 0.0
            ? lower_alpha + (upper_alpha - lower_alpha)
                * (target_time - lower_time) / denominator
            : 0.5 * (lower_alpha + upper_alpha);
        const double sine = std::sin(alpha);
        const double cosine = std::cos(alpha);
        const double radius = std::fma(
            geometry.apocenter - geometry.pericenter,
            sine * sine,
            geometry.pericenter);
        const double speed = std::sqrt(std::max(
            0.0,
            2.0 * (geometry.energy - effective_potential(radius))));
        // A Newton correction uses the physical time derivative when finite;
        // the time bracket and secant fallback keep the inverse inside its panel.
        const double dt_dalpha = speed > 0.0
            ? geometry.time_quadrature_scale
                * 2.0 * (geometry.apocenter - geometry.pericenter)
                * sine * cosine / speed
            : 0.0;
        if (dt_dalpha > 0.0 && std::isfinite(dt_dalpha)) {
            const double newton = alpha
                - (represented_time - target_time) / dt_dalpha;
            if (newton > lower_alpha && newton < upper_alpha) {
                next_alpha = newton;
            }
        }
        alpha = std::clamp(next_alpha, lower_alpha, upper_alpha);
        const double time_scale = std::max(
            1.0, geometry.visible_outgoing_time);
        if (upper_time - lower_time
            <= 64.0 * std::numeric_limits<double>::epsilon() * time_scale) {
            break;
        }
    }
    const double sine = std::sin(alpha);
    const double radius = std::fma(
        geometry.apocenter - geometry.pericenter,
        sine * sine,
        geometry.pericenter);
    const double speed = std::sqrt(std::max(
        0.0,
        2.0 * (geometry.energy - effective_potential(radius))));
    return {{radius, outgoing ? speed : -speed}, true, 0.0};
}

/// Uses the regularized monotone time table without fresh orbit integrals.
/// The table is built in alpha, where r=rp+(ra-rp)sin^2(alpha), so linear
/// interpolation remains well behaved at both turning points.  The returned
/// radial speed is recomputed from H-Phi_eff(r), which preserves the orbit
/// energy relation even though the phase inversion is interpolated.
template <class EffectivePotential>
inline HamiltonianOrbitPhaseGeometry::Location
hamiltonian_orbit_location_at_phase_table(
    const HamiltonianOrbitPhaseGeometry& geometry,
    double phase,
    EffectivePotential&& effective_potential) {
    phase = hamiltonian_orbit_wrap_phase(phase);
    const double physical_time = phase * geometry.radial_period;
    // The final part of a full phase cycle belongs to the unresolved
    // central excursion, rather than a fictitious resolved-grid position.
    if (physical_time > 2.0 * geometry.visible_outgoing_time) {
        return {
            {geometry.pericenter, 0.0},
            false,
            geometry.radial_period - physical_time,
        };
    }
    const bool outgoing = physical_time <= geometry.visible_outgoing_time;
    const double target_time = outgoing
        ? physical_time
        : 2.0 * geometry.visible_outgoing_time - physical_time;
    if (target_time <= 0.0) {
        const double boundary_speed = std::sqrt(std::max(
            0.0,
            2.0 * (geometry.energy
                - effective_potential(geometry.pericenter))));
        return {{
            geometry.pericenter,
            outgoing ? boundary_speed : -boundary_speed,
        }, true, 0.0};
    }
    if (target_time >= geometry.visible_outgoing_time) {
        return {{geometry.apocenter, 0.0}, true, 0.0};
    }
    if (geometry.alpha_edge.size() < 2
        || geometry.outgoing_time_edge.size() != geometry.alpha_edge.size()) {
        throw std::runtime_error(
            "Hamiltonian orbit has no consistent regularized time table.");
    }
    const auto upper = std::lower_bound(
        geometry.outgoing_time_edge.begin(),
        geometry.outgoing_time_edge.end(),
        target_time);
    std::size_t right = static_cast<std::size_t>(
        upper - geometry.outgoing_time_edge.begin());
    right = std::clamp<std::size_t>(
        right, 1, geometry.outgoing_time_edge.size() - 1);
    const std::size_t left = right - 1;
    // The table-only path interpolates alpha within the monotone travel-time
    // panel, then derives speed from the Hamiltonian at the returned radius.
    const double time_width = geometry.outgoing_time_edge[right]
        - geometry.outgoing_time_edge[left];
    const double fraction = time_width > 0.0
        ? std::clamp(
            (target_time - geometry.outgoing_time_edge[left]) / time_width,
            0.0,
            1.0)
        : 0.5;
    const double alpha = std::fma(
        fraction,
        geometry.alpha_edge[right] - geometry.alpha_edge[left],
        geometry.alpha_edge[left]);
    const double sine = std::sin(alpha);
    const double radius = std::fma(
        geometry.apocenter - geometry.pericenter,
        sine * sine,
        geometry.pericenter);
    const double speed = std::sqrt(std::max(
        0.0,
        2.0 * (geometry.energy - effective_potential(radius))));
    return {{radius, outgoing ? speed : -speed}, true, 0.0};
}

template <class EffectivePotential>
// Convert an actual orbit point back to directed travel phase. At the
// outer zero-velocity point choose the shared apocenter phase consistently.
inline double hamiltonian_orbit_phase_of_state(
    const HamiltonianOrbitPhaseGeometry& geometry,
    const CharacteristicState& state,
    EffectivePotential&& effective_potential) {
    const double radial_span = geometry.apocenter - geometry.pericenter;
    const double radial_fraction = std::clamp(
        (state.r - geometry.pericenter) / radial_span, 0.0, 1.0);
    const double alpha = std::asin(std::sqrt(radial_fraction));
    const double outgoing_time = hamiltonian_orbit_outgoing_time_at_alpha(
        geometry, alpha, effective_potential);
    double physical_time = outgoing_time;
    if (state.w < 0.0) {
        physical_time = 2.0 * geometry.visible_outgoing_time - outgoing_time;
    } else if (state.w == 0.0
               && state.r > 0.5
                    * (geometry.pericenter + geometry.apocenter)) {
        physical_time = geometry.visible_outgoing_time;
    }
    return hamiltonian_orbit_wrap_phase(
        physical_time / geometry.radial_period);
}

template <class EffectivePotential>
// Build the reusable regularized time table for one orbit, then locate
// the requested uniform phase-cell centers for atlas construction.
inline HamiltonianOrbitPhaseGeometry build_hamiltonian_orbit_phase_geometry(
    double eta,
    double energy,
    const std::vector<double>& scan_radius,
    int phase_cells,
    EffectivePotential&& effective_potential,
    double hidden_return_time =
        std::numeric_limits<double>::quiet_NaN()) {
    if (!(eta >= 0.0) || !std::isfinite(eta) || !std::isfinite(energy)
        || phase_cells < 4) {
        throw std::runtime_error("Invalid Hamiltonian orbit-phase request.");
    }
    bool inner_is_turning_point = true;
    std::array<double, 2> turning{};
    const double inner_value = effective_potential(scan_radius.front()) - energy;
    if (inner_value < 0.0 && std::isfinite(hidden_return_time)
        && hidden_return_time >= 0.0) {
        inner_is_turning_point = false;
        turning[0] = scan_radius.front();
        bool found_outer = false;
        for (std::size_t q = 0; q + 1 < scan_radius.size(); ++q) {
            const double left_value =
                effective_potential(scan_radius[q]) - energy;
            const double right_value =
                effective_potential(scan_radius[q + 1]) - energy;
            if (left_value <= 0.0 && right_value >= 0.0) {
                turning[1] = hamiltonian_orbit_bisect_turning_point(
                    scan_radius[q],
                    scan_radius[q + 1],
                    energy,
                    effective_potential);
                found_outer = true;
                break;
            }
        }
        if (!found_outer) {
            throw std::runtime_error(
                "Inner-crossing orbit has no represented outer turning point.");
        }
    } else {
        turning = hamiltonian_orbit_find_turning_points(
            scan_radius, energy, effective_potential);
        hidden_return_time = 0.0;
    }
    // Combine uniform alpha panels with mapped radial mesh knots. This
    // resolves changes of the represented field at the same spatial faces.
    const int cumulative_panels = std::max(256, 8 * phase_cells);
    constexpr double half_pi = 0.5 * PI;
    HamiltonianOrbitPhaseGeometry geometry;
    geometry.eta = eta;
    geometry.energy = energy;
    geometry.pericenter = turning[0];
    geometry.apocenter = turning[1];
    geometry.inner_is_turning_point = inner_is_turning_point;
    geometry.alpha_edge.reserve(
        static_cast<std::size_t>(cumulative_panels + 1)
        + scan_radius.size());
    for (int panel = 0; panel <= cumulative_panels; ++panel) {
        geometry.alpha_edge.push_back(
            half_pi * static_cast<double>(panel)
            / static_cast<double>(cumulative_panels));
    }
    const double radial_span = turning[1] - turning[0];
    for (double radius : scan_radius) {
        if (!(radius > turning[0]) || !(radius < turning[1])) {
            continue;
        }
        const double radial_fraction = std::clamp(
            (radius - turning[0]) / radial_span, 0.0, 1.0);
        geometry.alpha_edge.push_back(
            std::asin(std::sqrt(radial_fraction)));
    }
    std::sort(geometry.alpha_edge.begin(), geometry.alpha_edge.end());
    geometry.alpha_edge.erase(
        std::unique(
            geometry.alpha_edge.begin(),
            geometry.alpha_edge.end(),
            [](double first, double second) {
                return std::abs(first - second)
                    <= 128.0 * std::numeric_limits<double>::epsilon()
                        * std::max({1.0, std::abs(first), std::abs(second)});
            }),
        geometry.alpha_edge.end());
    // Cumulative positive panel travel times make the table monotone and
    // support bounded phase inversion without a new full-orbit integration.
    geometry.outgoing_time_edge.assign(geometry.alpha_edge.size(), 0.0);
    for (std::size_t panel = 0; panel + 1 < geometry.alpha_edge.size(); ++panel) {
        geometry.outgoing_time_edge[panel + 1] =
            geometry.outgoing_time_edge[panel]
            + hamiltonian_orbit_time_integral_alpha(
                geometry.alpha_edge[panel],
                geometry.alpha_edge[panel + 1],
                1,
                geometry.pericenter,
                geometry.apocenter,
                geometry.energy,
                effective_potential,
                geometry.inner_is_turning_point);
    }
    const double half_period = geometry.outgoing_time_edge.back();
    if (!(half_period > 0.0) || !std::isfinite(half_period)) {
        throw std::runtime_error("Orbit radial period is invalid.");
    }
    geometry.visible_outgoing_time = half_period;
    geometry.hidden_return_time = hidden_return_time;
    geometry.radial_period = 2.0 * half_period + hidden_return_time;
    geometry.time_quadrature_scale = 1.0;
    // Store physical or hidden-delay locations at each phase-cell center;
    // transport later advances an offset while this frozen geometry is valid.
    geometry.phase_center.resize(static_cast<std::size_t>(phase_cells));
    for (int q = 0; q < phase_cells; ++q) {
        geometry.phase_center[static_cast<std::size_t>(q)] =
            hamiltonian_orbit_location_at_phase_cached(
                geometry,
                (static_cast<double>(q) + 0.5)
                    / static_cast<double>(phase_cells),
                effective_potential);
    }
    return geometry;
}

}  // namespace ev
