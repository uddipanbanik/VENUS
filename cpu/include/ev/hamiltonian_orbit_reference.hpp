#pragma once

// Construct an instantaneous orbit-averaged reference for Hamiltonian transport.
// Inputs are the current cell distribution and reconstruction, the frozen
// potential, phase-space geometry, and eta=L^2/2 quadrature. Outputs are a
// stationary reference and a residual whose sum is the original distribution.
// At fixed eta, H=w^2/2+Phi+eta/r^2 is the orbital energy and dr*dw=dH*d(tau),
// where tau is travel time along an orbit. A narrow-energy-band phase-area
// average therefore estimates the phase-mixed part of the current state.
// The code admits closed represented orbits, constructs F(H,eta), reconstructs
// it on cells, and scales it to leave a nonnegative residual. Updating the
// field or collision state requires rebuilding this instantaneous reference.
// The sum of reference and residual remains the physical distribution.

#include "ev/conservative_sldg_transport.hpp"

namespace ev {

// One eta plane stores a piecewise-linear energy profile. Its support is
// bounded by represented energies and by the threshold for leaving the domain.
struct HamiltonianOrbitReferenceProfile {
    double minimum_energy = 0.0;
    double maximum_energy = 0.0;
    double bin_width = 0.0;
    double closed_orbit_energy_limit =
        -std::numeric_limits<double>::infinity();
    std::vector<double> value;

    bool active() const {
        return value.size() >= 2 && bin_width > 0.0
            && maximum_energy > minimum_energy;
    }

    // Interpolate bin-center values only inside the closed-orbit energy
    // range. Open or unrepresented energies contribute zero reference.
    double evaluate(double energy) const {
        if (!active() || energy < minimum_energy
            || energy > maximum_energy
            || energy >= closed_orbit_energy_limit) {
            return 0.0;
        }
        const double coordinate =
            (energy - minimum_energy) / bin_width - 0.5;
        if (coordinate <= 0.0) {
            return value.front();
        }
        const double last = static_cast<double>(value.size() - 1);
        if (coordinate >= last) {
            return value.back();
        }
        const int left = static_cast<int>(std::floor(coordinate));
        const double fraction = coordinate - static_cast<double>(left);
        return (1.0 - fraction) * value[static_cast<std::size_t>(left)]
            + fraction * value[static_cast<std::size_t>(left + 1)];
    }
};

// Keep both polynomial parts and their cell averages so transport and
// diagnostics use the same exact algebraic split of the reconstructed total.
struct HamiltonianOrbitReference {
    std::vector<double> cell_average;
    std::vector<SLDGSourcePolynomial> polynomial;
    std::vector<SLDGSourcePolynomial> residual_polynomial;
    std::vector<double> residual_cell_average;
    std::vector<HamiltonianOrbitReferenceProfile> profile;
    // Residual P2 polynomials at each eta Gauss node, flattened as
    // q_eta * cell_count + cell.  The ordinary residual_polynomial array is
    // their eta-cell average and remains useful for diagnostics/restarts.
    std::vector<SLDGSourcePolynomial> eta_node_residual_polynomial;
    int eta_quadrature_points = 1;
    double physical_reference_mass = 0.0;
    double physical_residual_l1 = 0.0;
    double physical_distribution_l1 = 0.0;
    double reference_mass_fraction = 0.0;
    double residual_l1_fraction = 0.0;
    double minimum_positive_reference_scale = 1.0;
    long long positive_reference_scaled_planes = 0;
    long long active_energy_planes = 0;
};

// This is specific Hamiltonian energy for a static frozen spherical field.
// The angular term eta/r^2 equals the tangential kinetic energy per unit mass.
inline double hamiltonian_orbit_reference_energy(
    double radius,
    double radial_velocity,
    double eta,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    const Params& p) {
    const double phi = sample_vp_field(radius, x_r, r, field, p).phi;
    return 0.5 * radial_velocity * radial_velocity
        + phi + eta / (radius * radius);
}

// A stationary orbit reference needs trajectories contained in the represented
// radial/velocity domain. Open radial faces supply finite effective-potential
// barriers; reflecting walls remove their escape-energy restriction.
inline double hamiltonian_orbit_reference_closed_energy_limit(
    double eta,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    const double inner_radius = geometry.radial.edge.front();
    const double outer_radius = geometry.radial.edge.back();
    auto effective_potential = [&](double radius) {
        return sample_vp_field(radius, x_r, r, field, p).phi
            + eta / (radius * radius);
    };
    const double inner_barrier = p.inner_boundary == "reflecting"
        ? std::numeric_limits<double>::infinity()
        : effective_potential(inner_radius);
    const double outer_barrier = p.outer_boundary == "reflecting"
        ? std::numeric_limits<double>::infinity()
        : effective_potential(outer_radius);
    double minimum_effective = std::numeric_limits<double>::infinity();
    for (double radius : geometry.radial.edge) {
        minimum_effective = std::min(
            minimum_effective, effective_potential(radius));
    }
    for (double radius : geometry.radial.center) {
        minimum_effective = std::min(
            minimum_effective, effective_potential(radius));
    }
    // The finite velocity grid supplies an additional represented-energy
    // limit. Leave a rounding margin before treating an orbit as closed.
    const double velocity_barrier = minimum_effective
        + 0.5 * p.w_max * p.w_max;
    const double raw_limit = std::min({
        inner_barrier,
        outer_barrier,
        velocity_barrier,
    });
    const double margin = 256.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, std::abs(raw_limit));
    return raw_limit - margin;
}

/// Returns true only when every point in one (r,w) source cell belongs to a
/// closed orbit in the represented domain.  The stationary reference is kept
/// out of cells cut by an open-boundary separatrix, so their full positive
/// distribution can enter the delayed carrier or leave through an open face.
inline bool hamiltonian_orbit_reference_cell_is_closed(
    int i,
    int j,
    double eta,
    double closed_energy_limit,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p) {
    const double radial_lower =
        geometry.radial.edge[static_cast<std::size_t>(i)];
    const double radial_upper =
        geometry.radial.edge[static_cast<std::size_t>(i + 1)];
    const double velocity_lower =
        geometry.radial_velocity.edge[static_cast<std::size_t>(j)];
    const double velocity_upper =
        geometry.radial_velocity.edge[static_cast<std::size_t>(j + 1)];
    const double maximum_radial_kinetic = 0.5 * std::max(
        velocity_lower * velocity_lower,
        velocity_upper * velocity_upper);
    auto effective_potential = [&](double radius) {
        return sample_vp_field(radius, x_r, r, field, p).phi
            + eta / (radius * radius);
    };
    const double maximum_effective = std::max(
        effective_potential(radial_lower),
        effective_potential(radial_upper));
    const double maximum_energy =
        maximum_radial_kinetic + maximum_effective;
    const double margin = 256.0 * std::numeric_limits<double>::epsilon()
        * std::max({
            1.0,
            std::abs(maximum_energy),
            std::abs(closed_energy_limit),
        });
    return maximum_energy < closed_energy_limit - margin;
}

// Fill quadrature-empty energy bins from neighboring represented averages.
// A plane with no measured phase area has no usable stationary profile.
inline void hamiltonian_orbit_reference_fill_empty_bins(
    const std::vector<double>& denominator,
    std::vector<double>& value) {
    const int count = static_cast<int>(value.size());
    std::vector<int> valid;
    valid.reserve(value.size());
    for (int q = 0; q < count; ++q) {
        if (denominator[static_cast<std::size_t>(q)] > 0.0) {
            valid.push_back(q);
        }
    }
    if (valid.empty()) {
        value.clear();
        return;
    }
    for (int q = 0; q < valid.front(); ++q) {
        value[static_cast<std::size_t>(q)] =
            value[static_cast<std::size_t>(valid.front())];
    }
    for (std::size_t segment = 0; segment + 1 < valid.size(); ++segment) {
        const int left = valid[segment];
        const int right = valid[segment + 1];
        for (int q = left + 1; q < right; ++q) {
            const double fraction = static_cast<double>(q - left)
                / static_cast<double>(right - left);
            value[static_cast<std::size_t>(q)] =
                (1.0 - fraction) * value[static_cast<std::size_t>(left)]
                + fraction * value[static_cast<std::size_t>(right)];
        }
    }
    for (int q = valid.back() + 1; q < count; ++q) {
        value[static_cast<std::size_t>(q)] =
            value[static_cast<std::size_t>(valid.back())];
    }
}

// Subtract the scaled reference coefficient by coefficient, preserving
// total=reference+residual in the same polynomial basis and cell geometry.
inline SLDGSourcePolynomial hamiltonian_orbit_reference_residual(
    const SLDGSourcePolynomial& total,
    const SLDGSourcePolynomial& stationary,
    double stationary_scale) {
    SLDGSourcePolynomial residual = total;
    residual.cell_average -= stationary_scale * stationary.cell_average;
    for (std::size_t mode = 0; mode < residual.mode.size(); ++mode) {
        residual.mode[mode] -= stationary_scale * stationary.mode[mode];
    }
    return residual;
}

// Find the largest common reference fraction whose residual stays
// nonnegative everywhere in each reconstructed source cell on this eta plane.
inline double hamiltonian_orbit_reference_positive_plane_scale(
    int k,
    const std::vector<SLDGSourcePolynomial>& total,
    const std::vector<SLDGSourcePolynomial>& stationary,
    const Params& p) {
    std::vector<std::size_t> active_cells;
    active_cells.reserve(static_cast<std::size_t>(p.Nr) * p.Nw);
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            const std::size_t cell = idx(i, j, k, p);
            const SLDGSourcePolynomial& value = stationary[cell];
            double scale = std::abs(value.cell_average);
            for (double coefficient : value.mode) {
                scale = std::max(scale, std::abs(coefficient));
            }
            if (scale > 0.0) {
                active_cells.push_back(cell);
            }
        }
    }
    if (active_cells.empty()) {
        return 1.0;
    }
    // Use the polynomial's complete parent-cell minimum to keep the entire
    // reconstructed residual nonnegative.
    auto admissible = [&](double scale) {
        for (std::size_t cell : active_cells) {
            if (!(hamiltonian_orbit_reference_residual(
                      total[cell], stationary[cell], scale)
                      .minimum_on_parent_cell()
                  >= 0.0)) {
                return false;
            }
        }
        return true;
    };
    if (admissible(1.0)) {
        return 1.0;
    }
    if (!admissible(0.0)) {
        throw std::runtime_error(
            "Hamiltonian reconstruction supplied a negative total polynomial "
            "to the positive orbit-reference split.");
    }
    // Bisection retains a known admissible lower endpoint. One rounding
    // step toward zero leaves the returned fraction on its safe side.
    double lower = 0.0;
    double upper = 1.0;
    for (int iteration = 0; iteration < 52; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        if (admissible(middle)) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    return lower > 0.0 ? std::nextafter(lower, 0.0) : 0.0;
}

// Rebuild the energy reference from the current frozen field and total
// polynomial. Eta planes are independent, allowing disjoint threaded work.
inline HamiltonianOrbitReference build_hamiltonian_orbit_reference(
    const std::vector<double>& f,
    const std::vector<SLDGSourcePolynomial>& total_polynomial,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    const std::vector<double>* eta_override = nullptr) {
    if (f.size() != total_polynomial.size()) {
        throw std::runtime_error(
            "Hamiltonian orbit reference received incompatible cell arrays.");
    }
    if (eta_override != nullptr
        && eta_override->size() != static_cast<std::size_t>(p.NL)) {
        throw std::runtime_error(
            "Hamiltonian orbit reference eta override has the wrong size.");
    }
    HamiltonianOrbitReference reference;
    reference.cell_average.assign(f.size(), 0.0);
    reference.polynomial.resize(f.size());
    reference.residual_polynomial = total_polynomial;
    reference.residual_cell_average = f;
    reference.profile.resize(static_cast<std::size_t>(p.NL));
    const auto& rule = phase_space_gauss_legendre_rule(
        p.vp_hamiltonian_reference_quadrature_points);
    const int energy_bins = p.vp_hamiltonian_reference_energy_bins;

    struct PlaneDiagnostic {
        long double reference_mass = 0.0L;
        long double residual_l1 = 0.0L;
        long double distribution_l1 = 0.0L;
        double positive_reference_scale = 1.0;
        bool positive_reference_scaled = false;
        bool active = false;
    };
    std::vector<PlaneDiagnostic> plane(
        static_cast<std::size_t>(p.NL));
    const int threads = effective_threads(p, p.NL);
    parallel_for_dynamic_chunks(
        0,
        p.NL,
        threads,
        1,
        [&](int k_begin, int k_end, int) {
            for (int k = k_begin; k < k_end; ++k) {
                const double eta = eta_override != nullptr
                    ? (*eta_override)[static_cast<std::size_t>(k)]
                    : geometry.eta.center[static_cast<std::size_t>(k)];
                const double eta_tolerance = 128.0
                    * std::numeric_limits<double>::epsilon()
                    * std::max(
                        1.0,
                        geometry.eta.edge[static_cast<std::size_t>(k + 1)]);
                if (eta < geometry.eta.edge[static_cast<std::size_t>(k)]
                        - eta_tolerance
                    || eta > geometry.eta.edge[static_cast<std::size_t>(k + 1)]
                        + eta_tolerance) {
                    throw std::runtime_error(
                        "Hamiltonian orbit-reference eta node lies outside its cell.");
                }
                HamiltonianOrbitReferenceProfile profile;
                profile.closed_orbit_energy_limit =
                    hamiltonian_orbit_reference_closed_energy_limit(
                        eta, x_r, r, field, geometry, p);
                // Distinguish occupied energies from all quadrature-accessible
                // closed energies, then pad the occupied range within the latter.
                double minimum_energy = std::numeric_limits<double>::infinity();
                double maximum_energy = -std::numeric_limits<double>::infinity();
                double available_minimum_energy =
                    std::numeric_limits<double>::infinity();
                double available_maximum_energy =
                    -std::numeric_limits<double>::infinity();

                for (int i = 0; i < p.Nr; ++i) {
                    const double rc = geometry.radial.center[
                        static_cast<std::size_t>(i)];
                    const double hr = 0.5 * geometry.radial.width[
                        static_cast<std::size_t>(i)];
                    for (int j = 0; j < p.Nw; ++j) {
                        const std::size_t cell = idx(i, j, k, p);
                        const double wc = geometry.radial_velocity.center[
                            static_cast<std::size_t>(j)];
                        const double hw = 0.5 * geometry.radial_velocity.width[
                            static_cast<std::size_t>(j)];
                        for (double xr : rule.first) {
                            for (double xw : rule.first) {
                                const double radius = rc + hr * xr;
                                const double velocity = wc + hw * xw;
                                const double energy =
                                    hamiltonian_orbit_reference_energy(
                                        radius,
                                        velocity,
                                        eta,
                                        x_r,
                                        r,
                                        field,
                                        p);
                                if (energy >= profile.closed_orbit_energy_limit) {
                                    continue;
                                }
                                available_minimum_energy = std::min(
                                    available_minimum_energy, energy);
                                available_maximum_energy = std::max(
                                    available_maximum_energy, energy);
                                const double value = total_polynomial[cell].value(
                                    radius, velocity);
                                if (value > 0.0) {
                                    minimum_energy = std::min(
                                        minimum_energy, energy);
                                    maximum_energy = std::max(
                                        maximum_energy, energy);
                                }
                            }
                        }
                    }
                }
                if (!(maximum_energy > minimum_energy)
                    || !std::isfinite(minimum_energy)
                    || !std::isfinite(maximum_energy)) {
                    reference.profile[static_cast<std::size_t>(k)] =
                        std::move(profile);
                    continue;
                }
                const double occupied_span = maximum_energy - minimum_energy;
                const double guard_padding = occupied_span
                    * 8.0 / static_cast<double>(std::max(8, energy_bins - 16));
                const double roundoff_padding =
                    32.0 * std::numeric_limits<double>::epsilon()
                    * std::max({
                        1.0,
                        std::abs(minimum_energy),
                        std::abs(maximum_energy),
                    });
                profile.minimum_energy = std::max(
                    available_minimum_energy,
                    minimum_energy - guard_padding - roundoff_padding);
                profile.maximum_energy = std::min(
                    std::min(
                        profile.closed_orbit_energy_limit,
                        available_maximum_energy),
                    maximum_energy + guard_padding + roundoff_padding);
                profile.bin_width =
                    (profile.maximum_energy - profile.minimum_energy)
                    / static_cast<double>(energy_bins);
                if (!(profile.bin_width > 0.0)) {
                    reference.profile[static_cast<std::size_t>(k)] =
                        std::move(profile);
                    continue;
                }
                // Accumulate integral(f*dr*dw) and integral(dr*dw) in each
                // energy band. Their ratio is its phase-area-weighted mean F.
                std::vector<double> numerator(
                    static_cast<std::size_t>(energy_bins), 0.0);
                std::vector<double> denominator(
                    static_cast<std::size_t>(energy_bins), 0.0);
                for (int i = 0; i < p.Nr; ++i) {
                    const double rc = geometry.radial.center[
                        static_cast<std::size_t>(i)];
                    const double hr = 0.5 * geometry.radial.width[
                        static_cast<std::size_t>(i)];
                    for (int j = 0; j < p.Nw; ++j) {
                        const std::size_t cell = idx(i, j, k, p);
                        const double wc = geometry.radial_velocity.center[
                            static_cast<std::size_t>(j)];
                        const double hw = 0.5 * geometry.radial_velocity.width[
                            static_cast<std::size_t>(j)];
                        for (std::size_t qr = 0; qr < rule.first.size(); ++qr) {
                            for (std::size_t qw = 0; qw < rule.first.size(); ++qw) {
                                const double radius = rc + hr * rule.first[qr];
                                const double velocity = wc + hw * rule.first[qw];
                                const double energy =
                                    hamiltonian_orbit_reference_energy(
                                        radius,
                                        velocity,
                                        eta,
                                        x_r,
                                        r,
                                        field,
                                        p);
                                if (energy < profile.minimum_energy
                                    || energy > profile.maximum_energy
                                    || energy >= profile.closed_orbit_energy_limit) {
                                    continue;
                                }
                                int bin = static_cast<int>(std::floor(
                                    (energy - profile.minimum_energy)
                                    / profile.bin_width));
                                bin = std::clamp(bin, 0, energy_bins - 1);
                                // Tensor Gauss weights carry the radial and
                                // velocity Jacobians of this source cell.
                                const double area_weight = hr * hw
                                    * rule.second[qr] * rule.second[qw];
                                const double value = total_polynomial[cell].value(
                                    radius, velocity);
                                denominator[static_cast<std::size_t>(bin)] +=
                                    area_weight;
                                numerator[static_cast<std::size_t>(bin)] +=
                                    area_weight * value;
                            }
                        }
                    }
                }
                profile.value.assign(
                    static_cast<std::size_t>(energy_bins), 0.0);
                for (int bin = 0; bin < energy_bins; ++bin) {
                    const double volume = denominator[
                        static_cast<std::size_t>(bin)];
                    if (volume > 0.0) {
                        profile.value[static_cast<std::size_t>(bin)] = std::max(
                            0.0,
                            numerator[static_cast<std::size_t>(bin)] / volume);
                    }
                }
                hamiltonian_orbit_reference_fill_empty_bins(
                    denominator, profile.value);
                if (!profile.active()) {
                    reference.profile[static_cast<std::size_t>(k)] =
                        std::move(profile);
                    continue;
                }
                // Interpolation between energy-bin centers changes its
                // quadrature mass slightly. Normalize the interpolated
                // profile to the mass accumulated in these energy bands.
                long double target_reference_mass = 0.0L;
                for (double value : numerator) {
                    target_reference_mass += value;
                }
                long double interpolated_reference_mass = 0.0L;
                for (int i = 0; i < p.Nr; ++i) {
                    const double rc = geometry.radial.center[
                        static_cast<std::size_t>(i)];
                    const double hr = 0.5 * geometry.radial.width[
                        static_cast<std::size_t>(i)];
                    for (int j = 0; j < p.Nw; ++j) {
                        const double wc = geometry.radial_velocity.center[
                            static_cast<std::size_t>(j)];
                        const double hw = 0.5 * geometry.radial_velocity.width[
                            static_cast<std::size_t>(j)];
                        for (std::size_t qr = 0; qr < rule.first.size(); ++qr) {
                            for (std::size_t qw = 0; qw < rule.first.size(); ++qw) {
                                const double radius = rc + hr * rule.first[qr];
                                const double velocity = wc + hw * rule.first[qw];
                                const double energy =
                                    hamiltonian_orbit_reference_energy(
                                        radius,
                                        velocity,
                                        eta,
                                        x_r,
                                        r,
                                        field,
                                        p);
                                // Tensor Gauss weights carry the radial and
                                // velocity Jacobians of this source cell.
                                const double area_weight = hr * hw
                                    * rule.second[qr] * rule.second[qw];
                                interpolated_reference_mass += area_weight
                                    * profile.evaluate(energy);
                            }
                        }
                    }
                }
                if (interpolated_reference_mass > 0.0L) {
                    const double normalization = static_cast<double>(
                        target_reference_mass / interpolated_reference_mass);
                    for (double& value : profile.value) {
                        value *= normalization;
                    }
                }
                reference.profile[static_cast<std::size_t>(k)] = profile;

                for (int i = 0; i < p.Nr; ++i) {
                    const double rc = geometry.radial.center[
                        static_cast<std::size_t>(i)];
                    const double dr = geometry.radial.width[
                        static_cast<std::size_t>(i)];
                    for (int j = 0; j < p.Nw; ++j) {
                        const std::size_t cell = idx(i, j, k, p);
                        // Keep the full original distribution in open-face
                        // guard cells and in cells cut by an escape boundary,
                        // where a stationary closed-orbit reference is unsuitable.
                        const bool touches_open_inner_guard =
                            p.inner_boundary != "reflecting" && i < 2;
                        const bool touches_open_outer_guard =
                            p.outer_boundary != "reflecting" && i >= p.Nr - 2;
                        const bool touches_open_velocity_guard =
                            p.momentum_boundary != "reflecting"
                            && (j < 2 || j >= p.Nw - 2);
                        const bool cell_is_closed =
                            hamiltonian_orbit_reference_cell_is_closed(
                                i,
                                j,
                                eta,
                                profile.closed_orbit_energy_limit,
                                x_r,
                                r,
                                field,
                                geometry,
                                p);
                        if (touches_open_inner_guard
                            || touches_open_outer_guard
                            || touches_open_velocity_guard
                            || !cell_is_closed) {
                            continue;
                        }
                        const double wc = geometry.radial_velocity.center[
                            static_cast<std::size_t>(j)];
                        const double dw = geometry.radial_velocity.width[
                            static_cast<std::size_t>(j)];
                        long double average = 0.0L;
                        std::array<long double, 5> moment{};
                        for (std::size_t qr = 0; qr < rule.first.size(); ++qr) {
                            for (std::size_t qw = 0; qw < rule.first.size(); ++qw) {
                                const double x = 0.5 * rule.first[qr];
                                const double y = 0.5 * rule.first[qw];
                                const double radius = rc + dr * x;
                                const double velocity = wc + dw * y;
                                const double energy =
                                    hamiltonian_orbit_reference_energy(
                                        radius,
                                        velocity,
                                        eta,
                                        x_r,
                                        r,
                                        field,
                                        p);
                                const double value = profile.evaluate(energy);
                                const long double normalized_weight =
                                    0.25L * rule.second[qr] * rule.second[qw];
                                // Project onto mean-free P2 basis modes on
                                // normalized cell coordinates. The separate
                                // cell average carries the conserved mass.
                                const std::array<long double, 5> basis{{
                                    x,
                                    y,
                                    x * x - 1.0L / 12.0L,
                                    x * y,
                                    y * y - 1.0L / 12.0L,
                                }};
                                average += normalized_weight * value;
                                for (std::size_t mode = 0; mode < 5; ++mode) {
                                    moment[mode] += normalized_weight
                                        * static_cast<long double>(value)
                                        * basis[mode];
                                }
                            }
                        }
                        SLDGSourcePolynomial reference_polynomial;
                        reference_polynomial.cell_average =
                            static_cast<double>(average);
                        reference_polynomial.center_r = rc;
                        reference_polynomial.center_w = wc;
                        reference_polynomial.scale_r = dr;
                        reference_polynomial.scale_w = dw;
                        // Divide projected moments by the known squared
                        // basis norms, then enforce polynomial positivity
                        // while retaining the reference cell average.
                        constexpr std::array<long double, 5> inverse_norm{{
                            12.0L, 12.0L, 180.0L, 144.0L, 180.0L}};
                        for (std::size_t mode = 0; mode < 5; ++mode) {
                            reference_polynomial.mode[mode] =
                                static_cast<double>(inverse_norm[mode] * moment[mode]);
                        }
                        reference_polynomial.enforce_positivity();
                        reference.cell_average[cell] =
                            reference_polynomial.cell_average;
                        reference.polynomial[cell] = reference_polynomial;
                    }
                }

                PlaneDiagnostic& diagnostic =
                    plane[static_cast<std::size_t>(k)];
                HamiltonianOrbitReferenceProfile& stored_profile =
                    reference.profile[static_cast<std::size_t>(k)];
                // A common eta-plane scale retains the energy-profile shape
                // while ensuring both installed polynomial parts are positive.
                const double positive_reference_scale =
                    hamiltonian_orbit_reference_positive_plane_scale(
                        k,
                        total_polynomial,
                        reference.polynomial,
                        p);
                diagnostic.positive_reference_scale = positive_reference_scale;
                diagnostic.positive_reference_scaled =
                    positive_reference_scale < 1.0;
                for (double& value : stored_profile.value) {
                    value *= positive_reference_scale;
                }
                for (int i = 0; i < p.Nr; ++i) {
                    const double rc = geometry.radial.center[
                        static_cast<std::size_t>(i)];
                    const double dr = geometry.radial.width[
                        static_cast<std::size_t>(i)];
                    for (int j = 0; j < p.Nw; ++j) {
                        const std::size_t cell = idx(i, j, k, p);
                        const double wc = geometry.radial_velocity.center[
                            static_cast<std::size_t>(j)];
                        const double dw = geometry.radial_velocity.width[
                            static_cast<std::size_t>(j)];
                        SLDGSourcePolynomial& stationary =
                            reference.polynomial[cell];
                        stationary.cell_average *= positive_reference_scale;
                        for (double& coefficient : stationary.mode) {
                            coefficient *= positive_reference_scale;
                        }
                        reference.cell_average[cell] = stationary.cell_average;
                        const SLDGSourcePolynomial residual =
                            hamiltonian_orbit_reference_residual(
                                total_polynomial[cell], stationary, 1.0);
                        if (!(residual.minimum_on_parent_cell() >= 0.0)) {
                            throw std::runtime_error(
                                "Positive orbit-reference split produced a "
                                "negative residual polynomial.");
                        }
                        reference.residual_cell_average[cell] =
                            residual.cell_average;
                        reference.residual_polynomial[cell] = residual;

                        // Integrate the reference mass and residual/total
                        // L1 measures using the same cell quadrature. These
                        // quantify how much physical variation remains to move.
                        const long double cell_area =
                            static_cast<long double>(dr) * dw;
                        diagnostic.reference_mass +=
                            cell_area * stationary.cell_average;
                        for (std::size_t qr = 0; qr < rule.first.size(); ++qr) {
                            for (std::size_t qw = 0; qw < rule.first.size(); ++qw) {
                                const double radius = rc
                                    + 0.5 * dr * rule.first[qr];
                                const double velocity = wc
                                    + 0.5 * dw * rule.first[qw];
                                const long double area_weight = 0.25L
                                    * cell_area * rule.second[qr]
                                    * rule.second[qw];
                                diagnostic.residual_l1 += area_weight
                                    * std::abs(residual.value(radius, velocity));
                                diagnostic.distribution_l1 += area_weight
                                    * std::abs(
                                        total_polynomial[cell].value(
                                            radius, velocity));
                            }
                        }
                    }
                }
                diagnostic.active = stored_profile.active()
                    && diagnostic.reference_mass > 0.0L;
            }
        });

    // Restore the spherical phase-space factor 8*pi^2*deta only when
    // combining independent eta-plane integrals into physical diagnostics.
    reference.minimum_positive_reference_scale = 1.0;
    for (int k = 0; k < p.NL; ++k) {
        const long double measure = 8.0L * PI * PI
            * geometry.eta.width[static_cast<std::size_t>(k)];
        const PlaneDiagnostic& diagnostic = plane[static_cast<std::size_t>(k)];
        reference.physical_reference_mass += static_cast<double>(
            measure * diagnostic.reference_mass);
        reference.physical_residual_l1 += static_cast<double>(
            measure * diagnostic.residual_l1);
        reference.physical_distribution_l1 += static_cast<double>(
            measure * diagnostic.distribution_l1);
        reference.minimum_positive_reference_scale = std::min(
            reference.minimum_positive_reference_scale,
            diagnostic.positive_reference_scale);
        reference.positive_reference_scaled_planes +=
            diagnostic.positive_reference_scaled ? 1 : 0;
        reference.active_energy_planes += diagnostic.active ? 1 : 0;
    }
    if (reference.physical_distribution_l1 > 0.0) {
        reference.reference_mass_fraction =
            reference.physical_reference_mass
            / reference.physical_distribution_l1;
        reference.residual_l1_fraction =
            reference.physical_residual_l1
            / reference.physical_distribution_l1;
    }
    return reference;
}

/// Builds a separate instantaneous orbit average at every eta quadrature node.
/// The stationary references and signed residuals are then averaged over the
/// eta cell with the same Gauss rule used by the characteristic transport.
inline HamiltonianOrbitReference
build_hamiltonian_orbit_reference_eta_quadrature(
    const std::vector<double>& f,
    const std::vector<SLDGSourcePolynomial>& total_polynomial,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const VlasovPoissonField& field,
    const PhaseSpaceCellGeometry& geometry,
    const Params& p,
    const std::vector<SLDGSourcePolynomial>* eta_node_total_polynomial = nullptr) {
    const auto& rule = phase_space_gauss_legendre_rule(
        p.vp_sldg_eta_quadrature_points);
    if (eta_node_total_polynomial != nullptr
        && eta_node_total_polynomial->size() != rule.first.size() * f.size()) {
        throw std::runtime_error(
            "Hamiltonian orbit reference received the wrong eta-node total array.");
    }
    HamiltonianOrbitReference averaged;
    averaged.cell_average.assign(f.size(), 0.0);
    averaged.polynomial.resize(f.size());
    averaged.residual_polynomial.resize(f.size());
    averaged.residual_cell_average.assign(f.size(), 0.0);
    averaged.eta_quadrature_points =
        static_cast<int>(rule.first.size());
    averaged.minimum_positive_reference_scale = 1.0;
    averaged.eta_node_residual_polynomial.resize(
        rule.first.size() * f.size());
    std::vector<unsigned char> active_plane(
        static_cast<std::size_t>(p.NL), 0);

    // Build the split at actual eta Gauss nodes before averaging. This
    // retains the angular-momentum dependence seen by characteristic transport.
    for (std::size_t q_eta = 0; q_eta < rule.first.size(); ++q_eta) {
        const double eta_weight = 0.5 * rule.second[q_eta];
        std::vector<double> eta_node(static_cast<std::size_t>(p.NL), 0.0);
        for (int k = 0; k < p.NL; ++k) {
            eta_node[static_cast<std::size_t>(k)] =
                geometry.eta.center[static_cast<std::size_t>(k)]
                + 0.5 * geometry.eta.width[static_cast<std::size_t>(k)]
                    * rule.first[q_eta];
        }
        std::vector<SLDGSourcePolynomial> node_total;
        std::vector<double> node_average;
        const std::vector<SLDGSourcePolynomial>* node_total_view =
            &total_polynomial;
        const std::vector<double>* node_average_view = &f;
        // Use node-resolved total polynomials when available; otherwise the
        // eta-cell polynomial supplies the common total at every node.
        if (eta_node_total_polynomial != nullptr) {
            const auto begin = eta_node_total_polynomial->begin()
                + static_cast<std::ptrdiff_t>(q_eta * f.size());
            node_total.assign(
                begin, begin + static_cast<std::ptrdiff_t>(f.size()));
            node_average.resize(f.size(), 0.0);
            for (std::size_t cell = 0; cell < f.size(); ++cell) {
                node_average[cell] = node_total[cell].cell_average;
            }
            node_total_view = &node_total;
            node_average_view = &node_average;
        }
        HamiltonianOrbitReference node = build_hamiltonian_orbit_reference(
            *node_average_view,
            *node_total_view,
            x_r,
            r,
            field,
            geometry,
            p,
            &eta_node);
        for (std::size_t cell = 0; cell < f.size(); ++cell) {
            averaged.cell_average[cell] +=
                eta_weight * node.cell_average[cell];
            averaged.residual_cell_average[cell] +=
                eta_weight * node.residual_cell_average[cell];
            SLDGSourcePolynomial& reference_polynomial =
                averaged.polynomial[cell];
            SLDGSourcePolynomial& residual_polynomial =
                averaged.residual_polynomial[cell];
            if (q_eta == 0) {
                reference_polynomial = node.polynomial[cell];
                residual_polynomial = node.residual_polynomial[cell];
                reference_polynomial.cell_average *= eta_weight;
                residual_polynomial.cell_average *= eta_weight;
                for (std::size_t mode = 0; mode < 5; ++mode) {
                    reference_polynomial.mode[mode] *= eta_weight;
                    residual_polynomial.mode[mode] *= eta_weight;
                }
            } else {
                reference_polynomial.cell_average +=
                    eta_weight * node.polynomial[cell].cell_average;
                residual_polynomial.cell_average +=
                    eta_weight * node.residual_polynomial[cell].cell_average;
                for (std::size_t mode = 0; mode < 5; ++mode) {
                    reference_polynomial.mode[mode] +=
                        eta_weight * node.polynomial[cell].mode[mode];
                    residual_polynomial.mode[mode] +=
                        eta_weight * node.residual_polynomial[cell].mode[mode];
                }
            }
            // Preserve each node residual for transport. The averaged modes
            // remain a separate cell-level diagnostic/restart representation.
            averaged.eta_node_residual_polynomial[
                q_eta * f.size() + cell] =
                node.residual_polynomial[cell];
        }
        averaged.physical_reference_mass +=
            eta_weight * node.physical_reference_mass;
        averaged.physical_residual_l1 +=
            eta_weight * node.physical_residual_l1;
        averaged.physical_distribution_l1 +=
            eta_weight * node.physical_distribution_l1;
        averaged.minimum_positive_reference_scale = std::min(
            averaged.minimum_positive_reference_scale,
            node.minimum_positive_reference_scale);
        averaged.positive_reference_scaled_planes +=
            node.positive_reference_scaled_planes;
        for (int k = 0; k < p.NL; ++k) {
            if (node.profile[static_cast<std::size_t>(k)].active()) {
                active_plane[static_cast<std::size_t>(k)] = 1;
            }
        }
    }
    averaged.active_energy_planes = static_cast<long long>(std::count(
        active_plane.begin(), active_plane.end(), static_cast<unsigned char>(1)));
    if (averaged.physical_distribution_l1 > 0.0) {
        averaged.reference_mass_fraction =
            averaged.physical_reference_mass
            / averaged.physical_distribution_l1;
        averaged.residual_l1_fraction =
            averaged.physical_residual_l1
            / averaged.physical_distribution_l1;
    }
    return averaged;
}

}  // namespace ev
