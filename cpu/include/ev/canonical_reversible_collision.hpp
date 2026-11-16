#pragma once

// Apply a local reversible Boltzmann increment to a canonical orbit atlas.
// The atlas is a table of orbit energy rows and phase cells. An explicit
// carrier retains a numerical mass element with its continuous product state.
// Inputs are the atlas, Eulerian collision geometry, current velocity grid,
// collision interval/law, and explicit open-mass and moment-repair budgets.
// Project the current state for the sampled or streamed collision calculation,
// then transfer only its before/after difference into stored phase cells.
// The preexisting orbit-carried distribution remains in its original atlas.
// Outputs are updated phase mass and energy moments, explicit open products,
// and physical/projection conservation and representation diagnostics.
// Staged mass positivity, energy-support, and conservation checks precede
// installation of the atlas increment and any newly created carrier output.

#include "ev/canonical_orbit_phase_transport.hpp"
#include "ev/sampled_reversible_boltzmann.hpp"
#include "ev/streamed_reversible_boltzmann.hpp"

namespace ev {

// Identify the largest collision increments outside the finite atlas so
// their physical radius, velocity, eta, and mass can be inspected together.
struct CanonicalEscapedIncrementCell {
    int radial_cell = -1;
    int radial_velocity_cell = -1;
    int eta_cell = -1;
    double reduced_mass = 0.0;
};

// Separate the physical collision budget from projection/representation
// errors, open mass, positivity clocks, and bounded moment repairs.
struct CanonicalReversibleCollisionDiagnostics {
    BPCollisionStepInfo collision;
    CanonicalProjectionLedger projection_before;
    CanonicalProjectionLedger projection_after;
    double physical_mass_before = 0.0;
    double physical_mass_after = 0.0;
    double physical_mass_residual = 0.0;
    double physical_radial_momentum_residual = 0.0;
    double physical_kinetic_energy_residual = 0.0;
    double mapped_physical_mass_residual = 0.0;
    double mapped_radial_momentum_residual = 0.0;
    double mapped_kinetic_energy_residual = 0.0;
    double atlas_mass_before = 0.0;
    double atlas_mass_after = 0.0;
    double atlas_mass_residual = 0.0;
    double escaped_increment_mass = 0.0;
    double escaped_positive_increment_mass = 0.0;
    double escaped_negative_increment_mass = 0.0;
    std::size_t escaped_positive_increment_cells = 0;
    std::size_t escaped_negative_increment_cells = 0;
    std::vector<CanonicalEscapedIncrementCell>
        largest_escaped_increment_cells;
    double represented_mass_residual = 0.0;
    double eulerian_increment_l1 = 0.0;
    double eulerian_increment_mass_l1 = 0.0;
    double atlas_increment_l1 = 0.0;
    double mapped_increment_defect_mass_l1 = 0.0;
    double minimum_atlas_cell_mass = std::numeric_limits<double>::infinity();
    double atlas_positivity_timescale =
        std::numeric_limits<double>::infinity();
    double atlas_mass_weighted_positivity_timescale =
        std::numeric_limits<double>::infinity();
    double positivity_repaired_mass = 0.0;
    std::size_t positivity_repaired_cells = 0;
    std::size_t positivity_repaired_rows = 0;
    double moment_repaired_mass = 0.0;
    double moment_repaired_energy_l1 = 0.0;
    double moment_repaired_energy2_l1 = 0.0;
    std::size_t moment_repaired_cells = 0;
    std::size_t moment_repaired_rows = 0;
};

// A losing cell of mass m and depletion time tau would develop a deficit
// m*max(h/tau-1,0) over interval h. Find h for which the sum of those
// deficits reaches the supplied absolute mass budget.
inline double canonical_mass_budgeted_positivity_timescale(
    const std::vector<double>& timescale,
    const std::vector<double>& cell_mass,
    double ignored_mass_budget) {
    if (timescale.size() != cell_mass.size()
        || !(ignored_mass_budget >= 0.0)
        || !std::isfinite(ignored_mass_budget)) {
        throw std::runtime_error(
            "Canonical positivity clock received invalid data.");
    }
    std::vector<std::size_t> order;
    order.reserve(timescale.size());
    for (std::size_t q = 0; q < timescale.size(); ++q) {
        if (!(cell_mass[q] >= 0.0) || !std::isfinite(cell_mass[q])
            || !(timescale[q] > 0.0) || std::isnan(timescale[q])) {
            throw std::runtime_error(
                "Canonical positivity clock received an invalid cell.");
        }
        if (std::isfinite(timescale[q])) {
            order.push_back(q);
        }
    }
    std::stable_sort(
        order.begin(),
        order.end(),
        [&](std::size_t first, std::size_t second) {
            return timescale[first] < timescale[second];
        });
    // Sort breakpoints tau, then solve the linear deficit equation on each
    // interval between them. Infinite depletion times contribute no deficit.
    long double active_mass = 0.0L;
    long double deficit_slope = 0.0L;
    for (std::size_t rank = 0; rank < order.size(); ++rank) {
        const std::size_t q = order[rank];
        if (cell_mass[q] > 0.0) {
            active_mass += cell_mass[q];
            deficit_slope += cell_mass[q] / timescale[q];
        }
        if (!(deficit_slope > 0.0L)) {
            continue;
        }
        const long double candidate =
            (static_cast<long double>(ignored_mass_budget) + active_mass)
            / deficit_slope;
        const double next_limit = rank + 1 < order.size()
            ? timescale[order[rank + 1]]
            : std::numeric_limits<double>::infinity();
        if (candidate <= next_limit) {
            return static_cast<double>(std::max(
                candidate,
                static_cast<long double>(timescale[q])));
        }
    }
    return std::numeric_limits<double>::infinity();
}

// Repair only a row whose negative cells lie within the supplied absolute
// tolerance. Zero those cells and rescale positive masses to the original
// nonnegative row total, carrying their energy moments with the same scaling.
inline double canonical_repair_row_mass_positivity(
    CanonicalOrbitRow& row,
    double tolerance,
    std::size_t& repaired_cells) {
    long double target_mass = 0.0L;
    long double positive_mass = 0.0L;
    long double negative_deficit = 0.0L;
    repaired_cells = 0;
    for (double mass : row.phase_mass) {
        if (!std::isfinite(mass) || mass < -tolerance) {
            throw std::runtime_error(
                "Canonical row positivity repair received an invalid mass.");
        }
        target_mass += mass;
        if (mass < 0.0) {
            negative_deficit -= mass;
            ++repaired_cells;
        } else {
            positive_mass += mass;
        }
    }
    if (repaired_cells == 0) {
        return 0.0;
    }
    if (target_mass < -tolerance
        || !(positive_mass > 0.0L)
        || target_mass > positive_mass) {
        throw std::runtime_error(
            "Canonical row has no nonnegative mass realization.");
    }
    target_mass = std::max(0.0L, target_mass);
    const long double scale = target_mass / positive_mass;
    std::size_t largest = 0;
    for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
        if (row.phase_mass[q] <= 0.0) {
            row.phase_mass[q] = 0.0;
            row.phase_energy_mass[q] = 0.0;
            row.phase_energy2_mass[q] = 0.0;
            continue;
        }
        row.phase_mass[q] = static_cast<double>(
            static_cast<long double>(row.phase_mass[q]) * scale);
        row.phase_energy_mass[q] = static_cast<double>(
            static_cast<long double>(row.phase_energy_mass[q]) * scale);
        row.phase_energy2_mass[q] = static_cast<double>(
            static_cast<long double>(row.phase_energy2_mass[q]) * scale);
        if (row.phase_mass[q] > row.phase_mass[largest]) {
            largest = q;
        }
    }
    // Close the final rounded row sum on its largest cell, updating its
    // energy moments in proportion to the small residual mass correction.
    const long double repaired_sum = std::accumulate(
        row.phase_mass.begin(), row.phase_mass.end(), 0.0L);
    const long double residual = target_mass - repaired_sum;
    if (residual != 0.0L) {
        const double old_mass = row.phase_mass[largest];
        const double new_mass = old_mass + static_cast<double>(residual);
        if (!(new_mass >= 0.0) || !std::isfinite(new_mass)) {
            throw std::runtime_error(
                "Canonical row positivity repair failed its residual update.");
        }
        const double moment_scale = old_mass > 0.0
            ? new_mass / old_mass
            : 0.0;
        row.phase_mass[largest] = new_mass;
        row.phase_energy_mass[largest] *= moment_scale;
        row.phase_energy2_mass[largest] *= moment_scale;
    }
    return static_cast<double>(negative_deficit);
}

// Physical collision invariants are mass, radial momentum, and total
// radial-plus-tangential kinetic energy, integrated over spherical volume.
struct CanonicalCollisionMoments {
    long double mass = 0.0L;
    long double radial_momentum = 0.0L;
    long double kinetic_energy = 0.0L;
};

inline CanonicalCollisionMoments canonical_collision_moments(
    const std::vector<double>& f,
    const PhaseSpaceCellGeometry& geometry,
    const std::vector<double>& radius,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p) {
    const Moments moments = compute_vp_moments(
        f, radius, w, L, wq, Lq, p);
    CanonicalCollisionMoments result;
    for (int i = 0; i < p.Nr; ++i) {
        const long double measure = 4.0L * PI
            * geometry.radial.center[static_cast<std::size_t>(i)]
            * geometry.radial.center[static_cast<std::size_t>(i)]
            * geometry.radial.width[static_cast<std::size_t>(i)];
        result.mass += measure * moments.rho[static_cast<std::size_t>(i)];
        result.radial_momentum +=
            measure * moments.j[static_cast<std::size_t>(i)];
        result.kinetic_energy += 0.5L * measure * (
            moments.p[static_cast<std::size_t>(i)]
            + moments.pt[static_cast<std::size_t>(i)]);
    }
    return result;
}

// Collision increments are located in current physical phase. Apply the
// inverse accumulated phase shift before adding them to the stored row,
// preserving its total increment across periodic cell overlaps.
inline std::vector<double> canonical_inverse_phase_shift_increment(
    const CanonicalOrbitRow& row,
    const std::vector<double>& physical_increment) {
    if (physical_increment.size() != row.phase_mass.size()) {
        throw std::runtime_error(
            "Canonical collision phase increment has the wrong size.");
    }
    CanonicalOrbitRow inverse = row;
    inverse.phase_offset = hamiltonian_orbit_wrap_phase(-row.phase_offset);
    std::vector<double> stored_increment(physical_increment.size(), 0.0);
    for (std::size_t q = 0; q < physical_increment.size(); ++q) {
        stored_increment[q] = canonical_orbit_geometric_cell_quantity(
            inverse, physical_increment, q);
    }
    const long double source_sum = std::accumulate(
        physical_increment.begin(), physical_increment.end(), 0.0L);
    const long double stored_sum = std::accumulate(
        stored_increment.begin(), stored_increment.end(), 0.0L);
    const long double residual = source_sum - stored_sum;
    if (!stored_increment.empty() && residual != 0.0L) {
        const auto destination = std::max_element(
            stored_increment.begin(),
            stored_increment.end(),
            [](double first, double second) {
                return std::abs(first) < std::abs(second);
            });
        *destination += static_cast<double>(residual);
    }
    return stored_increment;
}

// Move cell moment values into their allowed lower/upper bounds while
// preserving their row sum. A shared weighted offset followed by clamping
// makes the constrained sum monotone and suitable for bracketed bisection.
inline double canonical_conservative_box_projection(
    std::vector<double>& value,
    const std::vector<double>& lower,
    const std::vector<double>& upper,
    const std::vector<double>& weight) {
    if (value.size() != lower.size() || value.size() != upper.size()
        || value.size() != weight.size()) {
        throw std::runtime_error(
            "Canonical moment projection received incompatible vectors.");
    }
    const std::vector<double> original = value;
    long double target = 0.0L;
    long double lower_sum = 0.0L;
    long double upper_sum = 0.0L;
    double lambda_low = std::numeric_limits<double>::infinity();
    double lambda_high = -std::numeric_limits<double>::infinity();
    for (std::size_t q = 0; q < value.size(); ++q) {
        if (!std::isfinite(value[q]) || !std::isfinite(lower[q])
            || !std::isfinite(upper[q]) || !(weight[q] >= 0.0)
            || !std::isfinite(weight[q]) || lower[q] > upper[q]) {
            throw std::runtime_error(
                "Canonical moment projection received invalid bounds.");
        }
        target += value[q];
        lower_sum += lower[q];
        upper_sum += upper[q];
        if (weight[q] > 0.0) {
            lambda_low = std::min(
                lambda_low, (lower[q] - value[q]) / weight[q]);
            lambda_high = std::max(
                lambda_high, (upper[q] - value[q]) / weight[q]);
        }
    }
    const long double scale = std::max({
        1.0L,
        std::abs(target),
        std::abs(lower_sum),
        std::abs(upper_sum),
    });
    const long double tolerance = 16384.0L
        * std::numeric_limits<double>::epsilon() * scale;
    if (target < lower_sum - tolerance
        || target > upper_sum + tolerance) {
        throw std::runtime_error(
            "Canonical row moments have no nonnegative realization.");
    }
    // The conserved target must fit inside the summed support interval.
    // Clamp only its stated rounding-sized excursion before solving.
    target = std::max(lower_sum, std::min(upper_sum, target));
    if (!std::isfinite(lambda_low) || !std::isfinite(lambda_high)) {
        for (std::size_t q = 0; q < value.size(); ++q) {
            value[q] = std::clamp(value[q], lower[q], upper[q]);
        }
        double l1 = 0.0;
        for (std::size_t q = 0; q < value.size(); ++q) {
            l1 += std::abs(value[q] - original[q]);
        }
        return l1;
    }
    for (int iteration = 0; iteration < 96; ++iteration) {
        const double lambda = 0.5 * (lambda_low + lambda_high);
        long double sum = 0.0L;
        for (std::size_t q = 0; q < value.size(); ++q) {
            sum += std::clamp(
                std::fma(lambda, weight[q], value[q]),
                lower[q],
                upper[q]);
        }
        if (sum < target) {
            lambda_low = lambda;
        } else {
            lambda_high = lambda;
        }
    }
    const double lambda = 0.5 * (lambda_low + lambda_high);
    long double sum = 0.0L;
    for (std::size_t q = 0; q < value.size(); ++q) {
        value[q] = std::clamp(
            std::fma(lambda, weight[q], value[q]),
            lower[q],
            upper[q]);
        sum += value[q];
    }
    // Distribute the remaining rounded sum error only into available bound
    // room, retaining both the conserved target and cellwise admissibility.
    long double residual = target - sum;
    for (std::size_t q = 0; q < value.size() && residual != 0.0L; ++q) {
        const long double room = residual > 0.0L
            ? static_cast<long double>(upper[q] - value[q])
            : static_cast<long double>(value[q] - lower[q]);
        const long double correction = std::copysign(
            std::min(std::abs(residual), std::max(0.0L, room)),
            residual);
        value[q] += static_cast<double>(correction);
        residual -= correction;
    }
    if (std::abs(residual) > tolerance) {
        throw std::runtime_error(
            "Canonical moment projection failed its conservation ledger.");
    }
    double l1 = 0.0;
    for (std::size_t q = 0; q < value.size(); ++q) {
        l1 += std::abs(value[q] - original[q]);
    }
    return l1;
}

// A distribution supported on [H_left,H_right] has its mean in that
// interval and its second moment between mean^2 and the endpoint-mixture
// bound. Repair a bounded mass of cells while preserving row H and H^2 sums.
inline void canonical_repair_row_energy_moments(
    CanonicalOrbitRow& row,
    double mass_budget,
    CanonicalReversibleCollisionDiagnostics& diagnostics) {
    const std::size_t cells = row.phase_mass.size();
    std::vector<double> energy_lower(cells, 0.0);
    std::vector<double> energy_upper(cells, 0.0);
    std::vector<double> weight(cells, 0.0);
    double repair_mass = 0.0;
    std::size_t repair_cells = 0;
    const double energy_scale = std::max({
        1.0,
        std::abs(row.energy_left),
        std::abs(row.energy_right),
    });
    const double mean_tolerance = 8192.0
        * std::numeric_limits<double>::epsilon() * energy_scale;
    for (std::size_t q = 0; q < cells; ++q) {
        const double mass = row.phase_mass[q];
        weight[q] = mass;
        energy_lower[q] = mass * row.energy_left;
        energy_upper[q] = mass * row.energy_right;
        bool invalid = !std::isfinite(row.phase_energy_mass[q])
            || !std::isfinite(row.phase_energy2_mass[q]);
        if (mass > 0.0 && !invalid) {
            const double mean = row.phase_energy_mass[q] / mass;
            const double variance_numerator =
                row.phase_energy2_mass[q] * mass
                - row.phase_energy_mass[q] * row.phase_energy_mass[q];
            const double maximum_second_moment =
                (row.energy_left + row.energy_right)
                    * row.phase_energy_mass[q]
                - row.energy_left * row.energy_right * mass;
            const double variance_tolerance = 8192.0
                * std::numeric_limits<double>::epsilon()
                * std::max(
                    1.0,
                    std::abs(row.phase_energy2_mass[q] * mass));
            invalid = mean < row.energy_left - mean_tolerance
                || mean > row.energy_right + mean_tolerance
                || variance_numerator < -variance_tolerance
                || row.phase_energy2_mass[q]
                    > maximum_second_moment + variance_tolerance;
        }
        if (mass == 0.0
            && (row.phase_energy_mass[q] != 0.0
                || row.phase_energy2_mass[q] != 0.0)) {
            invalid = true;
        }
        if (invalid) {
            repair_mass += mass;
            ++repair_cells;
        }
    }
    if (repair_cells == 0) {
        return;
    }
    // Treat the affected mass as a numerical-resolution budget. A repair
    // beyond that budget stops before changing the staged energy moments.
    if (repair_mass > mass_budget) {
        std::ostringstream message;
        message << "Canonical collision increment requires an H-moment "
                << "positivity repair above budget: mass=" << repair_mass
                << ", budget=" << mass_budget
                << ", H_left=" << row.energy_left
                << ", H_right=" << row.energy_right;
        throw std::runtime_error(message.str());
    }
    diagnostics.moment_repaired_mass += repair_mass;
    diagnostics.moment_repaired_cells += repair_cells;
    ++diagnostics.moment_repaired_rows;
    const long double energy_before = std::accumulate(
        row.phase_energy_mass.begin(),
        row.phase_energy_mass.end(),
        0.0L);
    const long double energy2_before = std::accumulate(
        row.phase_energy2_mass.begin(),
        row.phase_energy2_mass.end(),
        0.0L);
    diagnostics.moment_repaired_energy_l1 +=
        canonical_conservative_box_projection(
            row.phase_energy_mass,
            energy_lower,
            energy_upper,
            weight);

    // After repairing the first energy moment, its value determines the
    // nonnegative-variance lower bound and bounded-support upper bound on H^2.
    std::vector<double> energy2_lower(cells, 0.0);
    std::vector<double> energy2_upper(cells, 0.0);
    for (std::size_t q = 0; q < cells; ++q) {
        const double mass = row.phase_mass[q];
        if (mass > 0.0) {
            const double energy_mass = row.phase_energy_mass[q];
            energy2_lower[q] = energy_mass * energy_mass / mass;
            energy2_upper[q] =
                (row.energy_left + row.energy_right) * energy_mass
                - row.energy_left * row.energy_right * mass;
            if (energy2_upper[q] < energy2_lower[q]) {
                energy2_upper[q] = energy2_lower[q];
            }
        }
    }
    diagnostics.moment_repaired_energy2_l1 +=
        canonical_conservative_box_projection(
            row.phase_energy2_mass,
            energy2_lower,
            energy2_upper,
            weight);

    // Verify the actual projected first/second moment totals, then check
    // every installed cell against its energy interval and variance bounds.
    const long double energy_after = std::accumulate(
        row.phase_energy_mass.begin(),
        row.phase_energy_mass.end(),
        0.0L);
    const long double energy2_after = std::accumulate(
        row.phase_energy2_mass.begin(),
        row.phase_energy2_mass.end(),
        0.0L);
    const long double conservation_scale = std::max({
        1.0L,
        std::abs(energy_before),
        std::abs(energy2_before),
    });
    const long double conservation_tolerance = 32768.0L
        * std::numeric_limits<double>::epsilon() * conservation_scale;
    if (std::abs(energy_after - energy_before) > conservation_tolerance
        || std::abs(energy2_after - energy2_before)
            > conservation_tolerance) {
        throw std::runtime_error(
            "Canonical H-moment repair failed its conservation ledger.");
    }

    for (std::size_t q = 0; q < cells; ++q) {
        const double mass = row.phase_mass[q];
        if (mass == 0.0) {
            if (row.phase_energy_mass[q] != 0.0
                || row.phase_energy2_mass[q] != 0.0) {
                throw std::runtime_error(
                    "Canonical H-moment repair left moments in an empty cell.");
            }
            continue;
        }
        const double energy_mass = row.phase_energy_mass[q];
        const double energy2_mass = row.phase_energy2_mass[q];
        const double lower_energy = mass * row.energy_left;
        const double upper_energy = mass * row.energy_right;
        const double lower_energy2 = energy_mass * energy_mass / mass;
        const double upper_energy2 =
            (row.energy_left + row.energy_right) * energy_mass
            - row.energy_left * row.energy_right * mass;
        const double tolerance = 32768.0
            * std::numeric_limits<double>::epsilon()
            * std::max({
                1.0,
                std::abs(lower_energy),
                std::abs(upper_energy),
                std::abs(lower_energy2),
                std::abs(upper_energy2),
            });
        if (energy_mass < lower_energy - tolerance
            || energy_mass > upper_energy + tolerance
            || energy2_mass < lower_energy2 - tolerance
            || energy2_mass > upper_energy2 + tolerance) {
            throw std::runtime_error(
                "Canonical H-moment repair failed its support bounds.");
        }
    }
}

// Compute a finite collision response, map its increment into a staged
// atlas, and commit only after explicit mass/open-support/repair checks.
inline CanonicalReversibleCollisionDiagnostics
apply_canonical_reversible_collision_increment(
    CanonicalOrbitAtlas& atlas,
    const PhaseSpaceCellGeometry& geometry,
    const std::vector<double>& radius,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    double dt,
    const Params& p,
    double maximum_escaped_increment_fraction = 0.0,
    double maximum_moment_repair_mass_fraction = 0.0,
    std::vector<WeightedPICMarker>* escaped_increment_markers = nullptr,
    std::vector<double>* represented_after_collision = nullptr,
    BPStreamedCollisionWorkspace* streamed_workspace = nullptr) {
    CanonicalReversibleCollisionDiagnostics diagnostics;
    if (!(dt >= 0.0) || !std::isfinite(dt)) {
        throw std::runtime_error(
            "Canonical reversible collision received an invalid interval.");
    }
    if (!(maximum_escaped_increment_fraction >= 0.0)
        || !std::isfinite(maximum_escaped_increment_fraction)) {
        throw std::runtime_error(
            "Canonical reversible collision received an invalid escaped-mass budget.");
    }
    if (!(maximum_moment_repair_mass_fraction >= 0.0)
        || !std::isfinite(maximum_moment_repair_mass_fraction)) {
        throw std::runtime_error(
            "Canonical reversible collision received an invalid moment-repair budget.");
    }
    if (dt == 0.0 || p.bp_collision_strength == 0.0) {
        return diagnostics;
    }
    if (!using_bp_sampled_reversible_boltzmann(p)
        && !using_bp_streamed_reversible_boltzmann(p)) {
        throw std::runtime_error(
            "Canonical collision increment requires a sampled or streamed "
            "reversible Boltzmann operator.");
    }

    // Observe the current distribution with its hidden moments. Both
    // collision states start from this same projection and fixed geometry.
    std::vector<double> before;
    std::vector<CanonicalHiddenCellMoments> hidden;
    canonical_orbit_project_to_eulerian(
        atlas, geometry, p, before, &hidden);
    std::vector<double> after = before;
    MpiContext mpi;
    diagnostics.collision = using_bp_streamed_reversible_boltzmann(p)
        ? apply_bp_streamed_reversible_boltzmann_collision_step(
            after,
            nullptr,
            radius,
            w,
            L,
            wq,
            Lq,
            dt,
            p,
            mpi,
            streamed_workspace)
        : apply_bp_sampled_reversible_boltzmann_collision_step(
            after, nullptr, radius, w, L, wq, Lq, dt, p, mpi);

    // Check mass, radial momentum, and kinetic energy at the physical
    // collision level before interpreting any atlas-projection difference.
    const CanonicalCollisionMoments before_moments =
        canonical_collision_moments(
            before, geometry, radius, w, L, wq, Lq, p);
    const CanonicalCollisionMoments after_moments =
        canonical_collision_moments(
            after, geometry, radius, w, L, wq, Lq, p);
    diagnostics.physical_mass_before = static_cast<double>(before_moments.mass);
    diagnostics.physical_mass_after = static_cast<double>(after_moments.mass);
    diagnostics.physical_mass_residual = static_cast<double>(
        after_moments.mass - before_moments.mass);
    diagnostics.physical_radial_momentum_residual = static_cast<double>(
        after_moments.radial_momentum - before_moments.radial_momentum);
    diagnostics.physical_kinetic_energy_residual = static_cast<double>(
        after_moments.kinetic_energy - before_moments.kinetic_energy);
    const double physical_mass_scale = std::max(
        1.0, std::abs(diagnostics.physical_mass_before));
    const double physical_momentum_scale = std::max(
        physical_mass_scale,
        std::abs(static_cast<double>(before_moments.radial_momentum)));
    const double physical_energy_scale = std::max(
        1.0, std::abs(static_cast<double>(before_moments.kinetic_energy)));
    if (std::abs(diagnostics.physical_mass_residual)
            > 2.0e-11 * physical_mass_scale
        || std::abs(diagnostics.physical_radial_momentum_residual)
            > 2.0e-11 * physical_momentum_scale
        || std::abs(diagnostics.physical_kinetic_energy_residual)
            > 2.0e-11 * physical_energy_scale) {
        std::ostringstream message;
        message << "Canonical reversible collision failed its physical "
                << "conservation ledger: dM="
                << diagnostics.physical_mass_residual
                << ", dPr="
                << diagnostics.physical_radial_momentum_residual
                << ", dK="
                << diagnostics.physical_kinetic_energy_residual;
        throw std::runtime_error(message.str());
    }
    for (std::size_t q = 0; q < before.size(); ++q) {
        diagnostics.eulerian_increment_l1 +=
            std::abs(after[q] - before[q]);
    }
    // Atlas overlaps define represented Eulerian cells. Products outside
    // this support require an explicit open carrier and their own mass ledger.
    std::vector<bool> atlas_supported(before.size(), false);
    for (const CanonicalOverlap& overlap : atlas.overlap) {
        if (overlap.eulerian_cell >= atlas_supported.size()) {
            throw std::runtime_error(
                "Canonical collision atlas overlap is out of range.");
        }
        atlas_supported[overlap.eulerian_cell] = true;
    }
    constexpr double physical_factor = 8.0 * PI * PI;
    std::vector<WeightedPICMarker> staged_escaped_increment_markers;
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t cell = idx(i, j, k, p);
                const double increment = (after[cell] - before[cell])
                    * canonical_cell_measure(geometry, i, j, k);
                diagnostics.eulerian_increment_mass_l1 +=
                    std::abs(increment);
                if (atlas_supported[cell]) {
                    continue;
                }
                // A positive unsupported increment creates physical carrier
                // mass. A negative one would require withdrawing an open
                // population that this atlas-only update does not contain.
                if (increment > 0.0) {
                    diagnostics.escaped_positive_increment_mass += increment;
                    ++diagnostics.escaped_positive_increment_cells;
                    staged_escaped_increment_markers.push_back(
                        WeightedPICMarker{
                            physical_factor * increment,
                            radius[static_cast<std::size_t>(i)],
                            w[static_cast<std::size_t>(j)],
                            L[static_cast<std::size_t>(k)],
                            0,
                            static_cast<std::uint64_t>(cell),
                            WeightedPICMarkerState::active,
                            0.0,
                            0.0,
                        });
                } else if (increment < 0.0) {
                    diagnostics.escaped_negative_increment_mass += increment;
                    ++diagnostics.escaped_negative_increment_cells;
                } else {
                    continue;
                }
                diagnostics.largest_escaped_increment_cells.push_back(
                    CanonicalEscapedIncrementCell{i, j, k, increment});
            }
        }
    }
    std::sort(
        diagnostics.largest_escaped_increment_cells.begin(),
        diagnostics.largest_escaped_increment_cells.end(),
        [](const CanonicalEscapedIncrementCell& first,
           const CanonicalEscapedIncrementCell& second) {
            return std::abs(first.reduced_mass)
                > std::abs(second.reduced_mass);
        });
    if (diagnostics.largest_escaped_increment_cells.size() > 8) {
        diagnostics.largest_escaped_increment_cells.resize(8);
    }

    // Project before and after onto identical atlas topology. Subtraction
    // cancels the common projection of the uncollided distribution, leaving
    // only the collision-induced change to transfer.
    CanonicalOrbitAtlas projected_before = atlas;
    CanonicalOrbitAtlas projected_after = atlas;
    diagnostics.projection_before =
        canonical_orbit_project_from_eulerian_conservative(
            projected_before, geometry, p, before, nullptr, &hidden);
    diagnostics.projection_after =
        canonical_orbit_project_from_eulerian_conservative(
            projected_after, geometry, p, after, nullptr, &hidden);
    diagnostics.escaped_increment_mass =
        diagnostics.projection_after.escaped_mass
        - diagnostics.projection_before.escaped_mass;
    const double escaped_cell_sum =
        diagnostics.escaped_positive_increment_mass
        + diagnostics.escaped_negative_increment_mass;
    const double escaped_cell_tolerance = 8192.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(diagnostics.physical_mass_before));
    if (std::abs(escaped_cell_sum - diagnostics.escaped_increment_mass)
        > escaped_cell_tolerance) {
        throw std::runtime_error(
            "Canonical collision escaped-cell ledger does not close.");
    }

    // Retain the original atlas until all increment budgets close. Apply
    // phase-unaligned differences to temporary row mass, H, and H^2 moments.
    CanonicalOrbitAtlas staged = atlas;
    diagnostics.atlas_mass_before = canonical_orbit_atlas_mass(atlas);
    std::vector<double> atlas_positivity_limit;
    std::vector<double> atlas_positivity_cell_mass;
    for (std::size_t row_index = 0; row_index < staged.row.size(); ++row_index) {
        CanonicalOrbitRow& destination = staged.row[row_index];
        const CanonicalOrbitRow& before_row = projected_before.row[row_index];
        const CanonicalOrbitRow& after_row = projected_after.row[row_index];
        if (destination.phase_mass.size() != before_row.phase_mass.size()
            || destination.phase_mass.size() != after_row.phase_mass.size()) {
            throw std::runtime_error(
                "Canonical collision increment changed atlas topology.");
        }
        std::vector<double> physical_mass_increment(
            destination.phase_mass.size(), 0.0);
        std::vector<double> physical_energy_increment(
            destination.phase_mass.size(), 0.0);
        std::vector<double> physical_energy2_increment(
            destination.phase_mass.size(), 0.0);
        for (std::size_t q = 0; q < destination.phase_mass.size(); ++q) {
            physical_mass_increment[q] =
                after_row.phase_mass[q] - before_row.phase_mass[q];
            physical_energy_increment[q] =
                after_row.phase_energy_mass[q]
                - before_row.phase_energy_mass[q];
            physical_energy2_increment[q] =
                after_row.phase_energy2_mass[q]
                - before_row.phase_energy2_mass[q];
        }
        // Undo the row's phase offset for each extensive moment. This puts
        // the physical collision response onto the stored phase coordinates.
        const std::vector<double> stored_mass_increment =
            canonical_inverse_phase_shift_increment(
                destination, physical_mass_increment);
        const std::vector<double> stored_energy_increment =
            canonical_inverse_phase_shift_increment(
                destination, physical_energy_increment);
        const std::vector<double> stored_energy2_increment =
            canonical_inverse_phase_shift_increment(
                destination, physical_energy2_increment);
        // Share one absolute mass budget between mass-positivity and
        // energy-support repairs, subtracting repairs already used by earlier rows.
        const double total_moment_repair_budget =
            maximum_moment_repair_mass_fraction
            * diagnostics.atlas_mass_before;
        const double remaining_repair_budget = std::max(
            0.0,
            total_moment_repair_budget
                - diagnostics.positivity_repaired_mass
                - diagnostics.moment_repaired_mass);
        const double roundoff_mass = 4096.0
            * std::numeric_limits<double>::epsilon()
            * std::max(1.0, diagnostics.atlas_mass_before);
        for (std::size_t q = 0; q < destination.phase_mass.size(); ++q) {
            if (stored_mass_increment[q] < 0.0
                && destination.phase_mass[q] > 0.0) {
                // A negative increment estimates this cell's depletion
                // time at the measured collision rate, independently of any
                // allowed small-mass repair of the proposed interval.
                const double positivity_limit =
                    dt * destination.phase_mass[q]
                    / (-stored_mass_increment[q]);
                diagnostics.atlas_positivity_timescale = std::min(
                    diagnostics.atlas_positivity_timescale,
                    positivity_limit);
                atlas_positivity_limit.push_back(positivity_limit);
                atlas_positivity_cell_mass.push_back(
                    destination.phase_mass[q]);
            }
            destination.phase_mass[q] += stored_mass_increment[q];
            destination.phase_energy_mass[q] += stored_energy_increment[q];
            destination.phase_energy2_mass[q] += stored_energy2_increment[q];
            diagnostics.atlas_increment_l1 +=
                std::abs(stored_mass_increment[q]);
            diagnostics.minimum_atlas_cell_mass = std::min(
                diagnostics.minimum_atlas_cell_mass,
                destination.phase_mass[q]);
            const double tolerance = std::max(
                roundoff_mass, remaining_repair_budget);
            if (destination.phase_mass[q] < -tolerance) {
                std::ostringstream message;
                message.precision(17);
                message
                    << "Canonical collision increment produced negative atlas mass"
                    << "; row=" << row_index
                    << ", phase_cell=" << q
                    << ", mass_before="
                    << destination.phase_mass[q] - stored_mass_increment[q]
                    << ", increment=" << stored_mass_increment[q]
                    << ", mass_after=" << destination.phase_mass[q]
                    << ", dt=" << dt
                    << ", estimated_positivity_dt="
                    << diagnostics.atlas_positivity_timescale
                    << ". Reduce the collision interval.";
                throw std::runtime_error(message.str());
            }
            if (!std::isfinite(destination.phase_energy_mass[q])
                || !std::isfinite(destination.phase_energy2_mass[q])) {
                throw std::runtime_error(
                    "Canonical collision increment produced non-finite H moments.");
            }
        }
        std::size_t positivity_repaired_cells = 0;
        const double positivity_repaired_mass =
            canonical_repair_row_mass_positivity(
                destination,
                std::max(roundoff_mass, remaining_repair_budget),
                positivity_repaired_cells);
        if (positivity_repaired_cells > 0) {
            diagnostics.positivity_repaired_mass +=
                positivity_repaired_mass;
            diagnostics.positivity_repaired_cells +=
                positivity_repaired_cells;
            ++diagnostics.positivity_repaired_rows;
        }
        if (diagnostics.positivity_repaired_mass
            > total_moment_repair_budget) {
            throw std::runtime_error(
                "Canonical collision positivity repair exceeded its mass budget.");
        }
        canonical_repair_row_energy_moments(
            destination,
            std::max(
                0.0,
                total_moment_repair_budget
                    - diagnostics.positivity_repaired_mass
                    - diagnostics.moment_repaired_mass),
            diagnostics);
    }
    // Report the mass-budgeted depletion time with the declared safety
    // fraction, alongside the strict minimum-cell positivity time.
    const double positivity_clock_budget = 0.25
        * maximum_moment_repair_mass_fraction
        * diagnostics.atlas_mass_before;
    diagnostics.atlas_mass_weighted_positivity_timescale =
        canonical_mass_budgeted_positivity_timescale(
            atlas_positivity_limit,
            atlas_positivity_cell_mass,
            positivity_clock_budget);

    // Re-observe the staged atlas to measure the mapped response error.
    // This diagnostic is distinct from the physical collision invariant check.
    std::vector<double> mapped_after;
    canonical_orbit_project_to_eulerian(
        staged, geometry, p, mapped_after);
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t cell = idx(i, j, k, p);
                if (atlas_supported[cell]) {
                    diagnostics.mapped_increment_defect_mass_l1 +=
                        std::abs(after[cell] - mapped_after[cell])
                        * canonical_cell_measure(geometry, i, j, k);
                }
            }
        }
    }
    diagnostics.atlas_mass_after = canonical_orbit_atlas_mass(staged);
    diagnostics.atlas_mass_residual =
        diagnostics.atlas_mass_after - diagnostics.atlas_mass_before;
    diagnostics.represented_mass_residual = diagnostics.atlas_mass_residual
        + diagnostics.escaped_increment_mass;
    const double mass_scale = std::max(1.0, diagnostics.atlas_mass_before);
    if (std::abs(diagnostics.represented_mass_residual)
        > 8192.0 * std::numeric_limits<double>::epsilon() * mass_scale) {
        std::ostringstream message;
        message << "Canonical collision increment failed its atlas mass ledger: "
                << "before=" << diagnostics.atlas_mass_before
                << ", after=" << diagnostics.atlas_mass_after
                << ", residual=" << diagnostics.atlas_mass_residual
                << ", projected_before_escaped="
                << diagnostics.projection_before.escaped_mass
                << ", projected_after_escaped="
                << diagnostics.projection_after.escaped_mass;
        throw std::runtime_error(message.str());
    }
    // Unsupported negative mass has no source ownership here. Positive
    // products require both an admitted open-mass budget and a carrier sink.
    const double escaped_roundoff = 8192.0
        * std::numeric_limits<double>::epsilon() * mass_scale;
    if (std::abs(diagnostics.escaped_negative_increment_mass)
        > escaped_roundoff) {
        throw std::runtime_error(
            "Canonical collision produced a negative increment outside "
            "the orbit atlas; the open distribution must participate in "
            "the collision update.");
    }
    if (std::abs(diagnostics.escaped_increment_mass)
        > maximum_escaped_increment_fraction
                * diagnostics.atlas_mass_before
            + escaped_roundoff) {
        std::ostringstream message;
        message << "Canonical collision increment exceeded its escaped-mass "
                << "budget: escaped_increment="
                << diagnostics.escaped_increment_mass << ", limit="
                << maximum_escaped_increment_fraction
                    * diagnostics.atlas_mass_before
                << ", positive="
                << diagnostics.escaped_positive_increment_mass
                << ", negative="
                << diagnostics.escaped_negative_increment_mass
                << ", positive_cells="
                << diagnostics.escaped_positive_increment_cells
                << ", negative_cells="
                << diagnostics.escaped_negative_increment_cells;
        for (const CanonicalEscapedIncrementCell& cell :
             diagnostics.largest_escaped_increment_cells) {
            message << " [i=" << cell.radial_cell
                    << ",j=" << cell.radial_velocity_cell
                    << ",k=" << cell.eta_cell
                    << ",r=" << radius[static_cast<std::size_t>(
                        cell.radial_cell)]
                    << ",w=" << w[static_cast<std::size_t>(
                        cell.radial_velocity_cell)]
                    << ",eta=" << L[static_cast<std::size_t>(
                        cell.eta_cell)]
                    << ",dm=" << cell.reduced_mass << ']';
        }
        throw std::runtime_error(message.str());
    }
    if (diagnostics.escaped_positive_increment_mass > escaped_roundoff
        && escaped_increment_markers == nullptr) {
        throw std::runtime_error(
            "Canonical collision produced open-orbit mass without a carrier sink.");
    }
    // Include new open products when reporting mapped physical moments,
    // so their kinetic energy and momentum remain in the represented budget.
    CanonicalCollisionMoments mapped_moments = canonical_collision_moments(
        mapped_after, geometry, radius, w, L, wq, Lq, p);
    for (const WeightedPICMarker& marker :
         staged_escaped_increment_markers) {
        mapped_moments.mass += marker.mass;
        mapped_moments.radial_momentum += marker.mass * marker.w;
        mapped_moments.kinetic_energy += marker.mass * (
            0.5 * marker.w * marker.w
            + marker.eta / (marker.r * marker.r));
    }
    diagnostics.mapped_physical_mass_residual = static_cast<double>(
        mapped_moments.mass - before_moments.mass);
    diagnostics.mapped_radial_momentum_residual = static_cast<double>(
        mapped_moments.radial_momentum - before_moments.radial_momentum);
    diagnostics.mapped_kinetic_energy_residual = static_cast<double>(
        mapped_moments.kinetic_energy - before_moments.kinetic_energy);
    // Install the checked staged atlas and return any explicit carrier and
    // Eulerian observation requested by the caller.
    atlas = std::move(staged);
    if (escaped_increment_markers != nullptr) {
        *escaped_increment_markers =
            std::move(staged_escaped_increment_markers);
    }
    if (represented_after_collision != nullptr) {
        *represented_after_collision = std::move(mapped_after);
    }
    return diagnostics;
}

}  // namespace ev
