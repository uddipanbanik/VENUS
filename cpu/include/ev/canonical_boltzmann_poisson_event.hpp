#pragma once

// Evolve a spherical Boltzmann-Poisson distribution represented by closed-orbit
// phase rows and explicit open collision products. The orbit atlas is a table
// of energy rows and phase cells. A carrier is a numerical mass element for
// an explicit collision product or an unresolved orbit segment. Inputs are
// that atlas, radial/velocity quadratures, the current Poisson field and
// collision parameters.
// The output state carries the new distribution, field, event ages and random
// streams; diagnostics close the represented mass and measure field changes.
//
// Within one frozen field, closed orbits advance by phase translation and open
// packets follow characteristics.  Collision events change the velocity
// distribution.  Field events deposit all represented components, solve Poisson
// and transfer the closed-orbit distribution to an atlas for the new potential.
// The event planner orders these operations at their physical deadlines.
// Potential and collision ages, together with random counters, are persistent
// state: restoring them preserves the elapsed exposure of a restarted run.

#include "ev/canonical_reversible_collision.hpp"
#include "ev/canonical_particle_ntc_collision.hpp"
#include "ev/canonical_orbit_phase_transport.hpp"
#include "ev/hamiltonian_sldg_transport.hpp"

namespace ev {

// The coupling selects how collision products are represented: continuous
// packets, signed distribution increments, or conservative atlas increments.
enum class CanonicalCollisionCoupling {
    carrier_ntc,
    delta_ntc,
    owned_ntc,
    batched_ntc,
    reversible_increment,
};

inline bool canonical_collision_retains_exact_carriers(
    CanonicalCollisionCoupling coupling) {
    return coupling == CanonicalCollisionCoupling::carrier_ntc
        || coupling == CanonicalCollisionCoupling::batched_ntc
        || coupling == CanonicalCollisionCoupling::delta_ntc;
}

// Fractions bound field variation, collision exposure and mass carried outside
// the closed-orbit/collision representation.  Intervals are in physical time.
struct CanonicalBoltzmannPoissonEventOptions {
    int energy_rows = 8;
    double maximum_collision_interval = 0.1;
    double potential_fraction = 0.05;
    double collision_fraction = 0.1;
    double collision_cfl_safety = 0.5;
    double maximum_open_mass_fraction = 1.0e-3;
    double maximum_collision_excluded_open_mass_fraction = 1.0e-3;
    double maximum_detached_mass_fraction = 1.0;
    // Exact post-collision carriers remain continuous phase-space packets
    // across a Poisson refresh.  Enabling this switch projects them back onto
    // the canonical atlas and is intended only for controlled remap tests.
    bool absorb_detached_carriers_on_field_rebuild = false;
    double potential_mass_fraction_floor = 1.0e-10;
    double startup_field_tolerance = 2.0e-2;
    int startup_field_iterations = 0;
    CanonicalCollisionCoupling collision_coupling =
        CanonicalCollisionCoupling::carrier_ntc;
    double maximum_collision_escaped_increment_fraction = 0.0;
    double maximum_collision_moment_repair_mass_fraction = 0.0;
    CanonicalAtlasBuildOptions atlas;
    CanonicalNTCCollisionOptions collision;
};

// The frozen field moves particles; poisson_reference records the density
// field at the last accepted atlas reconstruction for subsequent change tests.
struct CanonicalBoltzmannPoissonEventState {
    CanonicalOrbitAtlas atlas;
    CanonicalNTCCollisionState collision;
    BPStreamedCollisionWorkspace streamed_collision_workspace;
    VlasovPoissonField field;
    VlasovPoissonField poisson_reference;
    std::vector<double> eulerian_f;
    double time = 0.0;
    std::int64_t step = 0;
    double potential_age = 0.0;
    double collision_age = 0.0;
    double potential_timescale = std::numeric_limits<double>::infinity();
    double potential_deadline_age = std::numeric_limits<double>::infinity();
    double collision_timescale = std::numeric_limits<double>::infinity();
    double collision_majorant_timescale =
        std::numeric_limits<double>::infinity();
    double initial_physical_mass = 0.0;
    double hidden_physical_mass = 0.0;
    int startup_field_iterations = 0;
    double startup_field_mismatch = 0.0;
};

// Masses include the atlas factor 8*pi^2 and the physical packet weights.
// Hidden inner mass and escaped mass remain explicit in the total budget.
struct CanonicalBoltzmannPoissonEventDiagnostics {
    double dt = 0.0;
    WeightedPICEventPlan plan;
    bool collision_applied = false;
    double transport_mass_before = 0.0;
    double transport_mass_after = 0.0;
    double transport_mass_residual = 0.0;
    double collision_represented_mass_before = 0.0;
    double collision_represented_mass_after = 0.0;
    double collision_represented_mass_residual = 0.0;
    bool potential_rebuilt = false;
    CanonicalNTCCollisionDiagnostics collision;
    CanonicalSignedIncrementMaterializationDiagnostics
        signed_increment_materialization;
    CanonicalDetachedAbsorptionDiagnostics detached_absorption;
    CanonicalReversibleCollisionDiagnostics reversible_collision;
    double collision_seconds = 0.0;
    CanonicalProjectionLedger projection;
    CanonicalPreservedTopologyTransfer transfer;
    VPPotentialVariationLimit potential_variation;
    double atlas_build_seconds = 0.0;
    double physical_mass = 0.0;
    double open_mass = 0.0;
    double open_mass_fraction = 0.0;
    double active_open_mass = 0.0;
    double hidden_open_mass = 0.0;
    double escaped_open_mass = 0.0;
    double detached_mass = 0.0;
    double detached_mass_fraction = 0.0;
    double active_detached_mass = 0.0;
    double hidden_detached_mass = 0.0;
    double escaped_detached_mass = 0.0;
    std::size_t active_open_markers = 0;
    std::size_t hidden_open_markers = 0;
    std::size_t escaped_open_markers = 0;
    std::size_t active_detached_markers = 0;
    std::size_t hidden_detached_markers = 0;
    std::size_t escaped_detached_markers = 0;
    double open_transport_seconds = 0.0;
    double relative_potential_mismatch = 0.0;
    double relative_force_mismatch = 0.0;
    double mass_weighted_potential_mismatch = 0.0;
    double mass_weighted_force_mismatch = 0.0;
    double mass_weighted_potential_change = 0.0;
    double mass_weighted_force_change = 0.0;
};

// Local extrema locate large relative errors.  The event clock uses the
// mass-weighted RMS potential/force error returned by clock_norm().
struct CanonicalFieldMismatch {
    double potential = 0.0;
    double force = 0.0;
    double potential_mass_weighted_rms = 0.0;
    double force_mass_weighted_rms = 0.0;
    int potential_cell = -1;
    int force_cell = -1;
    double potential_cell_enclosed_mass_fraction = 0.0;
    double force_cell_enclosed_mass_fraction = 0.0;

    double maximum() const {
        return std::max(potential, force);
    }

    double clock_norm() const {
        return std::max(
            potential_mass_weighted_rms,
            force_mass_weighted_rms);
    }
};

// Validate fractions and finite positive intervals before they enter a clock
// division or an admissibility test.
inline void validate_canonical_boltzmann_poisson_event_options(
    const CanonicalBoltzmannPoissonEventOptions& options) {
    if (options.energy_rows < 1
        || !(options.maximum_collision_interval > 0.0)
        || !std::isfinite(options.maximum_collision_interval)
        || !(options.potential_fraction > 0.0)
        || !(options.potential_fraction <= 1.0)
        || !std::isfinite(options.potential_fraction)
        || !(options.collision_fraction > 0.0)
        || !(options.collision_fraction <= 1.0)
        || !std::isfinite(options.collision_fraction)
        || !(options.collision_cfl_safety > 0.0)
        || !(options.collision_cfl_safety <= 1.0)
        || !std::isfinite(options.collision_cfl_safety)
        || !(options.maximum_open_mass_fraction >= 0.0)
        || !std::isfinite(options.maximum_open_mass_fraction)
        || !(options.maximum_collision_excluded_open_mass_fraction >= 0.0)
        || !std::isfinite(
            options.maximum_collision_excluded_open_mass_fraction)
        || !(options.maximum_detached_mass_fraction > 0.0)
        || !(options.maximum_detached_mass_fraction <= 1.0)
        || !std::isfinite(options.maximum_detached_mass_fraction)
        || !(options.potential_mass_fraction_floor >= 0.0)
        || !(options.potential_mass_fraction_floor < 1.0)
        || !std::isfinite(options.potential_mass_fraction_floor)
        || !(options.startup_field_tolerance > 0.0)
        || !std::isfinite(options.startup_field_tolerance)
        || options.startup_field_iterations < 0
        || options.startup_field_iterations > 16
        || !(options.maximum_collision_escaped_increment_fraction >= 0.0)
        || !std::isfinite(
            options.maximum_collision_escaped_increment_fraction)
        || !(options.maximum_collision_moment_repair_mass_fraction >= 0.0)
        || !std::isfinite(
            options.maximum_collision_moment_repair_mass_fraction)) {
        throw std::runtime_error(
            "Invalid canonical Boltzmann-Poisson event options.");
    }
}

// Optionally project active detached packets back into closed-orbit rows using
// the same potential that currently defines those rows.
inline void absorb_canonical_collision_carriers(
    CanonicalBoltzmannPoissonEventState& state,
    const PhaseSpaceCellGeometry& geometry,
    const std::vector<double>& radial_coordinate,
    const std::vector<double>& radius,
    const Params& p,
    const CanonicalBoltzmannPoissonEventOptions& options,
    CanonicalBoltzmannPoissonEventDiagnostics& diagnostics) {
    if (!canonical_collision_retains_exact_carriers(
            options.collision_coupling)
        || state.collision.detached_marker.empty()) {
        return;
    }
    auto represented_potential = [&](double r) {
        return sample_vp_field(
            r, radial_coordinate, radius, state.field, p).phi;
    };
    diagnostics.detached_absorption =
        canonical_absorb_active_detached_markers(
            state.atlas,
            geometry,
            p,
            state.hidden_physical_mass,
            options.collision.maximum_lift_phase_interpolation_cell_error,
            options.collision.threads,
            state.collision,
            represented_potential);
}

// Compare two fields on the same density and exact spherical shell volumes.
// The enclosed-mass floor selects cells for both extrema and RMS errors.
// Shell masses weight the RMS norm, with hidden mass assigned to the first cell.
inline CanonicalFieldMismatch canonical_field_mismatch(
    const VlasovPoissonField& represented,
    const VlasovPoissonField& poisson,
    const Moments& moments,
    const PhaseSpaceCellGeometry& geometry,
    double hidden_mass,
    double mass_fraction_floor) {
    const std::size_t cells = geometry.radial.center.size();
    if (represented.phi.size() != cells || represented.phi_r.size() != cells
        || poisson.phi.size() != cells || poisson.phi_r.size() != cells
        || moments.rho.size() != cells) {
        throw std::runtime_error("Canonical field mismatch size mismatch.");
    }
    long double total_mass = hidden_mass;
    std::vector<long double> enclosed(cells, hidden_mass);
    std::vector<long double> shell_mass(cells, 0.0L);
    for (std::size_t i = 0; i < cells; ++i) {
        const double left = geometry.radial.edge[i];
        const double right = geometry.radial.edge[i + 1];
        const long double current_shell_mass = 4.0L * PI / 3.0L
            * (right * right * right - left * left * left)
            * std::max(0.0, moments.rho[i]);
        shell_mass[i] = current_shell_mass;
        total_mass += current_shell_mass;
        enclosed[i] = total_mass;
    }
    const double maximum_phi = std::max(
        *std::max_element(
            represented.phi.begin(), represented.phi.end(),
            [](double a, double b) { return std::abs(a) < std::abs(b); }),
        *std::max_element(
            poisson.phi.begin(), poisson.phi.end(),
            [](double a, double b) { return std::abs(a) < std::abs(b); }),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    const double maximum_force = std::max(
        *std::max_element(
            represented.phi_r.begin(), represented.phi_r.end(),
            [](double a, double b) { return std::abs(a) < std::abs(b); }),
        *std::max_element(
            poisson.phi_r.begin(), poisson.phi_r.end(),
            [](double a, double b) { return std::abs(a) < std::abs(b); }),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    const double phi_floor = 1.0e-12 * std::max(1.0, std::abs(maximum_phi));
    const double force_floor = 1.0e-12
        * std::max(1.0, std::abs(maximum_force));
    CanonicalFieldMismatch mismatch;
    long double supported_weight = 0.0L;
    long double potential_square = 0.0L;
    long double force_square = 0.0L;
    for (std::size_t i = 0; i < cells; ++i) {
        if (total_mass > 0.0L
            && enclosed[i] / total_mass < mass_fraction_floor) {
            continue;
        }
        const double potential_mismatch =
            std::abs(poisson.phi[i] - represented.phi[i])
                / std::max({
                    std::abs(poisson.phi[i]),
                    std::abs(represented.phi[i]),
                    phi_floor,
                });
        const long double weight = shell_mass[i]
            + (i == 0 ? static_cast<long double>(hidden_mass) : 0.0L);
        supported_weight += weight;
        potential_square += weight * potential_mismatch * potential_mismatch;
        if (potential_mismatch > mismatch.potential) {
            mismatch.potential = potential_mismatch;
            mismatch.potential_cell = static_cast<int>(i);
            mismatch.potential_cell_enclosed_mass_fraction = total_mass > 0.0L
                ? static_cast<double>(enclosed[i] / total_mass) : 0.0;
        }
        const double force_mismatch =
            std::abs(poisson.phi_r[i] - represented.phi_r[i])
                / std::max({
                    std::abs(poisson.phi_r[i]),
                    std::abs(represented.phi_r[i]),
                    force_floor,
                });
        force_square += weight * force_mismatch * force_mismatch;
        if (force_mismatch > mismatch.force) {
            mismatch.force = force_mismatch;
            mismatch.force_cell = static_cast<int>(i);
            mismatch.force_cell_enclosed_mass_fraction = total_mass > 0.0L
                ? static_cast<double>(enclosed[i] / total_mass) : 0.0;
        }
    }
    if (supported_weight > 0.0L) {
        mismatch.potential_mass_weighted_rms = std::sqrt(
            static_cast<double>(potential_square / supported_weight));
        mismatch.force_mass_weighted_rms = std::sqrt(
            static_cast<double>(force_square / supported_weight));
    }
    return mismatch;
}

// Estimate a static-field update time from continuity-derived dPhi/dt and
// d(Phi_r)/dt.  Each time is the ratio of mass-weighted field and rate norms.
// Resolved shell mass weights the numerator; total mass includes hidden mass.
inline VPPotentialVariationLimit
canonical_mass_weighted_potential_variation_limit(
    const Moments& moments,
    const VlasovPoissonField& field,
    const PhaseSpaceCellGeometry& geometry,
    const std::vector<double>& radius,
    double time,
    const Params& p,
    double hidden_mass) {
    VPPotentialVariationLimit limit;
    if (using_expanding_vp(p)) {
        return limit;
    }
    const VlasovPoissonField derivative = vp_potential_time_derivative(
        moments, radius, time, p);
    if (moments.rho.size() != radius.size()
        || geometry.radial.edge.size() != radius.size() + 1
        || field.phi.size() != radius.size()
        || derivative.phi.size() != radius.size()) {
        throw std::runtime_error(
            "Canonical potential clock size mismatch.");
    }
    long double total_mass = std::max(0.0, hidden_mass);
    std::vector<long double> weight(radius.size(), 0.0L);
    for (std::size_t i = 0; i < radius.size(); ++i) {
        const double left = geometry.radial.edge[i];
        const double right = geometry.radial.edge[i + 1];
        weight[i] = 4.0L * PI / 3.0L
            * (right * right * right - left * left * left)
            * std::max(0.0, moments.rho[i]);
        total_mass += weight[i];
    }
    if (!(total_mass > 0.0L)) {
        return limit;
    }
    long double phi2 = 0.0L;
    long double phi_rate2 = 0.0L;
    long double force2 = 0.0L;
    long double force_rate2 = 0.0L;
    double maximum_phi_rate_term = -1.0;
    double maximum_force_rate_term = -1.0;
    for (std::size_t i = 0; i < radius.size(); ++i) {
        const long double normalized_weight = weight[i] / total_mass;
        phi2 += normalized_weight * field.phi[i] * field.phi[i];
        phi_rate2 += normalized_weight
            * derivative.phi[i] * derivative.phi[i];
        force2 += normalized_weight
            * field.phi_r[i] * field.phi_r[i];
        force_rate2 += normalized_weight
            * derivative.phi_r[i] * derivative.phi_r[i];
        const double phi_term = static_cast<double>(normalized_weight)
            * derivative.phi[i] * derivative.phi[i];
        const double force_term = static_cast<double>(normalized_weight)
            * derivative.phi_r[i] * derivative.phi_r[i];
        if (phi_term > maximum_phi_rate_term) {
            maximum_phi_rate_term = phi_term;
            limit.controlling_potential_cell = static_cast<int>(i);
        }
        if (force_term > maximum_force_rate_term) {
            maximum_force_rate_term = force_term;
            limit.controlling_force_cell = static_cast<int>(i);
        }
    }
    const long double floor = 4096.0L
        * std::numeric_limits<double>::epsilon();
    if (phi2 > floor * floor && phi_rate2 > floor * floor) {
        limit.potential_timescale = static_cast<double>(
            std::sqrt(phi2 / phi_rate2));
    }
    if (force2 > floor * floor && force_rate2 > floor * floor) {
        limit.force_timescale = static_cast<double>(
            std::sqrt(force2 / force_rate2));
    }
    limit.timescale = std::min(
        limit.potential_timescale, limit.force_timescale);
    if (std::isfinite(limit.timescale)) {
        limit.timestep = p.vp_potential_variation_eta * limit.timescale;
    }
    return limit;
}

// Integrate dr/sqrt(2*(H-Phi-eta/r^2)) along a permitted radial segment.
// A quadratic radial substitution resolves the endpoint square-root behavior.
inline double canonical_open_radial_travel_time(
    double radius_begin,
    double radius_end,
    double energy,
    double eta,
    const std::function<double(double)>& potential) {
    if (!(radius_begin > 0.0) || !(radius_end > radius_begin)
        || !std::isfinite(energy) || !(eta >= 0.0)
        || !std::isfinite(eta)) {
        throw std::runtime_error("Invalid canonical open-orbit time integral.");
    }
    const auto& rule = phase_space_gauss_legendre_rule(12);
    constexpr int panels = 16;
    const double span = radius_end - radius_begin;
    long double integral = 0.0L;
    for (int panel = 0; panel < panels; ++panel) {
        const double x_left = static_cast<double>(panel) / panels;
        const double x_right = static_cast<double>(panel + 1) / panels;
        const double x_center = 0.5 * (x_left + x_right);
        const double x_half = 0.5 * (x_right - x_left);
        for (std::size_t q = 0; q < rule.first.size(); ++q) {
            const double x = x_center + x_half * rule.first[q];
            const double radius = radius_end - span * x * x;
            const double radial_speed2 = 2.0 * (
                energy - potential(radius) - eta / (radius * radius));
            const double scale = std::max(1.0, std::abs(energy));
            if (radial_speed2 < -2048.0
                    * std::numeric_limits<double>::epsilon() * scale) {
                throw std::runtime_error(
                    "Canonical open-orbit time path is dynamically forbidden.");
            }
            integral += x_half * rule.second[q] * 2.0 * span * x
                / std::sqrt(std::max(
                    radial_speed2, std::numeric_limits<double>::min()));
        }
    }
    return static_cast<double>(integral);
}

// Advance explicit products in the same frozen potential as the orbit atlas.
// Inner crossings enter the delayed-return representation; outer escape is
// recorded in the packet state so its mass remains in the boundary ledger.
inline void advance_canonical_open_orbit_carrier(
    std::vector<WeightedPICMarker>& marker,
    double dt,
    const std::vector<double>& radial_coordinate,
    const std::vector<double>& radius,
    const VlasovPoissonField& field,
    const Params& p,
    double hidden_physical_mass,
    unsigned int threads) {
    if (marker.empty() || dt == 0.0) {
        return;
    }
    Params orbit_params = p;
    orbit_params.inner_boundary = "reflecting";
    const CharacteristicIntegratorPlan plan =
        make_characteristic_integrator_plan(orbit_params);
    const double inner = p.r_min;
    const double outer = vp_radial_outer_boundary(radius, p);
    auto potential = [&](double value) {
        return sample_vp_field(
            value, radial_coordinate, radius, field, p).phi;
    };
    advance_weighted_pic_markers_parallel(
        marker,
        dt,
        [&](const WeightedPICMarker& source, double local_dt) {
            const double source_phi = potential(source.r);
            const double energy = source_phi
                + 0.5 * source.w * source.w
                + source.eta / (source.r * source.r);
            const double inner_effective = potential(inner)
                + source.eta / (inner * inner);
            if (source.w < 0.0 && energy > inner_effective) {
                const double crossing_time = source.r > inner
                    ? canonical_open_radial_travel_time(
                        inner, source.r, energy, source.eta, potential)
                    : 0.0;
                if (crossing_time <= local_dt) {
                    const double speed = std::sqrt(
                        2.0 * (energy - inner_effective));
                    const WeightedPICMarker at_face{
                        source.mass,
                        inner,
                        -speed,
                        source.eta,
                        source.id,
                        source.source_cell,
                        WeightedPICMarkerState::active,
                        0.0,
                        0.0,
                    };
                    return weighted_pic_delayed_inner_outcome(
                        at_face, crossing_time, hidden_physical_mass, p);
                }
            }
            if (source.w > 0.0) {
                const double outer_effective = potential(outer)
                    + source.eta / (outer * outer);
                if (energy > outer_effective) {
                    const double crossing_time = source.r < outer
                        ? canonical_open_radial_travel_time(
                            source.r, outer, energy, source.eta, potential)
                        : 0.0;
                    if (crossing_time <= local_dt) {
                        return WeightedPICOrbitOutcome{
                            WeightedPICOrbitDisposition::escaped,
                            outer,
                            std::sqrt(2.0 * (energy - outer_effective)),
                            crossing_time,
                            0.0,
                            0.0,
                        };
                    }
                }
            }
            const CharacteristicState advanced = advance_vp_characteristic(
                source.r,
                source.w,
                std::sqrt(2.0 * source.eta),
                radial_coordinate,
                radius,
                field,
                local_dt,
                orbit_params,
                plan);
            if (advanced.r > outer) {
                return WeightedPICOrbitOutcome{
                    WeightedPICOrbitDisposition::escaped,
                    outer,
                    std::abs(advanced.w),
                    local_dt,
                    0.0,
                    0.0,
                };
            }
            return WeightedPICOrbitOutcome{
                WeightedPICOrbitDisposition::active,
                advanced.r,
                advanced.w,
                local_dt,
                0.0,
                0.0,
            };
        },
        threads);
}

// Construct one Poisson source from atlas projection, open/detached packets and
// the positive-minus-negative signed increment, using a common radial mesh.
inline Moments canonical_boltzmann_poisson_event_moments(
    const CanonicalBoltzmannPoissonEventState& state,
    const PhaseSpaceCellGeometry& geometry,
    const std::vector<double>& radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const std::vector<double>& wq,
    const std::vector<double>& etaq,
    const Params& p,
    WeightedPICDepositedMoments* open_deposition = nullptr,
    WeightedPICDepositedMoments* detached_deposition = nullptr) {
    Moments moments = compute_vp_moments(
        state.eulerian_f, radius, w, eta, wq, etaq, p);
    const WeightedPICRadialMesh mesh = make_weighted_pic_radial_mesh(geometry);
    WeightedPICDepositedMoments deposited = deposit_weighted_vlasov_pic(
        state.collision.open_marker, mesh);
    WeightedPICDepositedMoments detached = deposit_weighted_vlasov_pic(
        state.collision.detached_marker, mesh);
    WeightedPICDepositedMoments signed_positive =
        deposit_weighted_vlasov_pic(
            state.collision.signed_increment_positive, mesh);
    WeightedPICDepositedMoments signed_negative =
        deposit_weighted_vlasov_pic(
            state.collision.signed_increment_negative, mesh);
    const Moments open = weighted_pic_as_venus_moments(deposited, mesh);
    const Moments detached_moments = weighted_pic_as_venus_moments(
        detached, mesh);
    const Moments signed_positive_moments = weighted_pic_as_venus_moments(
        signed_positive, mesh);
    const Moments signed_negative_moments = weighted_pic_as_venus_moments(
        signed_negative, mesh);
    for (std::size_t i = 0; i < moments.rho.size(); ++i) {
        moments.rho[i] += open.rho[i];
        moments.p[i] += open.p[i];
        moments.pt[i] += open.pt[i];
        moments.j[i] += open.j[i];
        moments.rho[i] += detached_moments.rho[i];
        moments.p[i] += detached_moments.p[i];
        moments.pt[i] += detached_moments.pt[i];
        moments.j[i] += detached_moments.j[i];
        moments.rho[i] += signed_positive_moments.rho[i]
            - signed_negative_moments.rho[i];
        moments.p[i] += signed_positive_moments.p[i]
            - signed_negative_moments.p[i];
        moments.pt[i] += signed_positive_moments.pt[i]
            - signed_negative_moments.pt[i];
        moments.j[i] += signed_positive_moments.j[i]
            - signed_negative_moments.j[i];
    }
    if (open_deposition != nullptr) {
        *open_deposition = std::move(deposited);
    }
    if (detached_deposition != nullptr) {
        *detached_deposition = std::move(detached);
    }
    return moments;
}

// These sums distinguish represented physical mass, active collision mass and
// signed increments.  Boundary state changes preserve a packet's mass weight.
inline double canonical_open_marker_mass(
    const CanonicalNTCCollisionState& collision) {
    long double mass = 0.0L;
    for (const WeightedPICMarker& marker : collision.open_marker) {
        mass += marker.mass;
    }
    return static_cast<double>(mass);
}

inline double canonical_detached_marker_mass(
    const CanonicalNTCCollisionState& collision) {
    long double mass = 0.0L;
    for (const WeightedPICMarker& marker : collision.detached_marker) {
        mass += marker.mass;
    }
    return static_cast<double>(mass);
}

inline double canonical_active_detached_marker_mass(
    const CanonicalNTCCollisionState& collision) {
    long double mass = 0.0L;
    for (const WeightedPICMarker& marker : collision.detached_marker) {
        if (marker.state == WeightedPICMarkerState::active) {
            mass += marker.mass;
        }
    }
    return static_cast<double>(mass);
}

inline double canonical_signed_marker_mass_by_state(
    const std::vector<WeightedPICMarker>& marker,
    WeightedPICMarkerState selected) {
    long double mass = 0.0L;
    for (const WeightedPICMarker& value : marker) {
        if (value.state == selected) {
            mass += value.mass;
        }
    }
    return static_cast<double>(mass);
}

inline double canonical_signed_increment_mass_by_state(
    const CanonicalNTCCollisionState& collision,
    WeightedPICMarkerState selected) {
    return canonical_signed_marker_mass_by_state(
        collision.signed_increment_positive, selected)
        - canonical_signed_marker_mass_by_state(
            collision.signed_increment_negative, selected);
}

inline double canonical_total_carrier_mass(
    const CanonicalNTCCollisionState& collision) {
    return canonical_open_marker_mass(collision)
        + canonical_detached_marker_mass(collision)
        + weighted_pic_total_marker_mass(
            collision.signed_increment_positive)
        - weighted_pic_total_marker_mass(
            collision.signed_increment_negative);
}

// Give newly created open products fresh persistent IDs before publication.
inline void canonical_append_open_collision_increment(
    CanonicalNTCCollisionState& collision,
    std::vector<WeightedPICMarker> marker) {
    collision.open_marker.reserve(
        collision.open_marker.size() + marker.size());
    for (WeightedPICMarker& value : marker) {
        if (collision.next_open_marker_id == 0) {
            throw std::runtime_error(
                "Canonical open collision marker id counter overflowed.");
        }
        value.id = collision.next_open_marker_id++;
        validate_weighted_pic_marker(
            value, "Canonical reversible open collision product");
        collision.open_marker.push_back(value);
    }
}

// The reversible atlas operator requires its excluded carrier mass to fit the
// configured budget and requires its supported carrier representation.
inline void validate_canonical_reversible_collision_carriers(
    const CanonicalBoltzmannPoissonEventState& state,
    const CanonicalBoltzmannPoissonEventOptions& options) {
    if (!state.collision.detached_marker.empty()
        || !state.collision.signed_increment_positive.empty()
        || !state.collision.signed_increment_negative.empty()) {
        throw std::runtime_error(
            "Reversible canonical collisions cannot consume NTC carriers.");
    }
    const double excluded_fraction = state.initial_physical_mass > 0.0
        ? canonical_open_marker_mass(state.collision)
            / state.initial_physical_mass
        : 0.0;
    if (excluded_fraction
        > options.maximum_collision_excluded_open_mass_fraction) {
        throw std::runtime_error(
            "Canonical open-orbit mass exceeded the reversible-collision "
            "exclusion budget.");
    }
}

// Initialize the projected density and its Poisson reference, check startup
// self-consistency, then seed the field/collision clocks for the first event.
inline CanonicalBoltzmannPoissonEventState
initialize_canonical_boltzmann_poisson_event_state(
    CanonicalOrbitAtlas atlas,
    VlasovPoissonField field,
    double time,
    std::int64_t step,
    const PhaseSpaceCellGeometry& geometry,
    const std::vector<double>& radial_coordinate,
    const std::vector<double>& radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const std::vector<double>& wq,
    const std::vector<double>& etaq,
    const Params& p,
    const CanonicalBoltzmannPoissonEventOptions& options) {
    validate_canonical_boltzmann_poisson_event_options(options);
    if (!(time >= 0.0) || !std::isfinite(time) || step < 0) {
        throw std::runtime_error(
            "Invalid canonical Boltzmann-Poisson initial event state.");
    }
    CanonicalBoltzmannPoissonEventState state;
    state.atlas = std::move(atlas);
    state.field = std::move(field);
    state.time = time;
    state.step = step;
    constexpr double physical_factor = 8.0 * PI * PI;
    const CanonicalProjectionLedger final_projection =
        canonical_orbit_project_to_eulerian(
            state.atlas, geometry, p, state.eulerian_f);
    state.hidden_physical_mass = physical_factor
        * final_projection.hidden_mass;
    const Moments moments = compute_vp_moments(
        state.eulerian_f, radius, w, eta, wq, etaq, p);
    state.poisson_reference = solve_vlasov_poisson_field(
        moments,
        radius,
        state.time,
        p,
        state.hidden_physical_mass);
    const CanonicalFieldMismatch mismatch = canonical_field_mismatch(
        state.field,
        state.poisson_reference,
        moments,
        geometry,
        state.hidden_physical_mass,
        options.potential_mass_fraction_floor);
    state.startup_field_mismatch = mismatch.clock_norm();
    if (state.startup_field_mismatch > options.startup_field_tolerance) {
        throw std::runtime_error(
            "Canonical startup representation exceeds its field tolerance; "
            "mass-weighted mismatch="
            + std::to_string(state.startup_field_mismatch)
            + ", potential mismatch=" + std::to_string(mismatch.potential)
            + ", mass-weighted potential mismatch="
            + std::to_string(mismatch.potential_mass_weighted_rms)
            + " at cell " + std::to_string(mismatch.potential_cell)
            + " (M(<r)/M="
            + std::to_string(
                mismatch.potential_cell_enclosed_mass_fraction) + ")"
            + ", force mismatch=" + std::to_string(mismatch.force)
            + ", mass-weighted force mismatch="
            + std::to_string(mismatch.force_mass_weighted_rms)
            + " at cell " + std::to_string(mismatch.force_cell)
            + " (M(<r)/M="
            + std::to_string(mismatch.force_cell_enclosed_mass_fraction)
            + "). Increase canonical resolution before evolution.");
    }
    (void)radial_coordinate;
    state.initial_physical_mass = physical_factor
        * final_projection.source_mass;
    state.hidden_physical_mass = physical_factor
        * final_projection.hidden_mass;
    const VPPotentialVariationLimit variation =
        canonical_mass_weighted_potential_variation_limit(
            moments,
            state.field,
            geometry,
            radius,
            state.time,
            p,
            state.hidden_physical_mass);
    state.potential_timescale = variation.timescale;
    state.potential_deadline_age = variation.timestep;
    // The first short event probes the collision distribution and replaces
    // both estimates with shell-resolved physical and majorant timescales.
    state.collision_timescale = options.maximum_collision_interval
        / options.collision_fraction;
    state.collision_majorant_timescale =
        options.maximum_collision_interval
        / options.collision_cfl_safety;
    return state;
}

// This entry applies a collision at every positive orbit interval and refreshes
// the field after the measured change reaches its prescribed fraction.
inline CanonicalBoltzmannPoissonEventDiagnostics
advance_canonical_boltzmann_poisson_event_legacy(
    CanonicalBoltzmannPoissonEventState& state,
    const PhaseSpaceCellGeometry& geometry,
    const std::vector<double>& radial_coordinate,
    const std::vector<double>& radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const std::vector<double>& wq,
    const std::vector<double>& etaq,
    const Params& p,
    const CanonicalBoltzmannPoissonEventOptions& options,
    double remaining_time = std::numeric_limits<double>::infinity()) {
    validate_canonical_boltzmann_poisson_event_options(options);
    if (options.collision_coupling
        == CanonicalCollisionCoupling::delta_ntc) {
        throw std::runtime_error(
            "Signed delta NTC requires the event-driven canonical advance.");
    }
    if (!(remaining_time > 0.0) || std::isnan(remaining_time)) {
        throw std::runtime_error(
            "Canonical Boltzmann-Poisson event needs positive remaining time.");
    }
    CanonicalBoltzmannPoissonEventDiagnostics diagnostics;
    diagnostics.dt = std::min(options.maximum_collision_interval, remaining_time);
    if (std::isfinite(state.collision_timescale)) {
        diagnostics.dt = std::min(diagnostics.dt, state.collision_timescale);
    }
    if (std::isfinite(state.potential_deadline_age)) {
        const double remaining_potential = std::max(
            0.0, state.potential_deadline_age - state.potential_age);
        diagnostics.dt = std::min(diagnostics.dt, remaining_potential);
    }
    if (!(diagnostics.dt > 0.0) || !std::isfinite(diagnostics.dt)) {
        throw std::runtime_error(
            "Canonical Boltzmann-Poisson event selected an invalid interval.");
    }

    const auto open_transport_begin = std::chrono::steady_clock::now();
    advance_canonical_open_orbit_carrier(
        state.collision.open_marker,
        diagnostics.dt,
        radial_coordinate,
        radius,
        state.field,
        p,
        state.hidden_physical_mass,
        options.collision.threads);
    advance_canonical_open_orbit_carrier(
        state.collision.detached_marker,
        diagnostics.dt,
        radial_coordinate,
        radius,
        state.field,
        p,
        state.hidden_physical_mass,
        options.collision.threads);
    advance_canonical_open_orbit_carrier(
        state.collision.signed_increment_positive,
        diagnostics.dt,
        radial_coordinate,
        radius,
        state.field,
        p,
        state.hidden_physical_mass,
        options.collision.threads);
    advance_canonical_open_orbit_carrier(
        state.collision.signed_increment_negative,
        diagnostics.dt,
        radial_coordinate,
        radius,
        state.field,
        p,
        state.hidden_physical_mass,
        options.collision.threads);
    diagnostics.open_transport_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - open_transport_begin).count();
    canonical_orbit_advance_frozen(state.atlas, diagnostics.dt);
    const auto collision_begin = std::chrono::steady_clock::now();
    constexpr double physical_factor = 8.0 * PI * PI;
    diagnostics.collision_represented_mass_before = physical_factor
        * canonical_orbit_atlas_mass(state.atlas)
        + canonical_total_carrier_mass(state.collision);
    if (options.collision_coupling
        == CanonicalCollisionCoupling::reversible_increment) {
        validate_canonical_reversible_collision_carriers(state, options);
        std::vector<WeightedPICMarker> open_increment;
        diagnostics.reversible_collision =
            apply_canonical_reversible_collision_increment(
                state.atlas,
                geometry,
                radius,
                w,
                eta,
                wq,
                etaq,
                diagnostics.dt,
                p,
                options.maximum_collision_escaped_increment_fraction,
                options.maximum_collision_moment_repair_mass_fraction,
                &open_increment,
                &state.eulerian_f);
        canonical_append_open_collision_increment(
            state.collision, std::move(open_increment));
        state.collision_timescale =
            diagnostics.reversible_collision.collision
                .collision_time_estimate;
        state.collision_majorant_timescale =
            std::numeric_limits<double>::infinity();
    } else {
        CanonicalNTCCollisionOptions collision_options = options.collision;
        collision_options.hidden_central_mass = state.hidden_physical_mass;
        collision_options.retain_collision_products_as_detached_carriers =
            options.collision_coupling
                != CanonicalCollisionCoupling::owned_ntc;
        auto potential = [&](double r) {
            return sample_vp_field(
                r, radial_coordinate, radius, state.field, p).phi;
        };
        diagnostics.collision = options.collision_coupling
                == CanonicalCollisionCoupling::owned_ntc
            ? apply_canonical_owned_ntc_collision(
                state.atlas,
                geometry,
                diagnostics.dt,
                p,
                collision_options,
                state.collision,
                potential)
            : apply_canonical_ntc_collision(
                state.atlas,
                geometry,
                diagnostics.dt,
                p,
                collision_options,
                state.collision,
                potential);
        state.collision_timescale = weighted_pic_collision_majorant_timescale(
            state.collision.workspace.shells);
    }
    diagnostics.collision_represented_mass_after = physical_factor
        * canonical_orbit_atlas_mass(state.atlas)
        + canonical_total_carrier_mass(state.collision);
    diagnostics.collision_represented_mass_residual =
        diagnostics.collision_represented_mass_after
        - diagnostics.collision_represented_mass_before;
    const double collision_mass_scale = std::max(
        1.0, diagnostics.collision_represented_mass_before);
    if (std::abs(diagnostics.collision_represented_mass_residual)
        > 2.0e-11 * collision_mass_scale) {
        std::ostringstream message;
        message.precision(17);
        message
            << "Canonical collision failed its represented total-mass ledger"
            << ": before="
            << diagnostics.collision_represented_mass_before
            << ", after="
            << diagnostics.collision_represented_mass_after
            << ", residual="
            << diagnostics.collision_represented_mass_residual
            << ", atlas_residual_reduced="
            << diagnostics.reversible_collision.atlas_mass_residual
            << ", escaped_increment_reduced="
            << diagnostics.reversible_collision.escaped_increment_mass
            << ", escaped_positive_physical="
            << physical_factor
                * diagnostics.reversible_collision
                    .escaped_positive_increment_mass
            << ".";
        throw std::runtime_error(message.str());
    }
    diagnostics.collision_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - collision_begin).count();
    diagnostics.collision_applied = true;
    state.time += diagnostics.dt;
    ++state.step;
    state.potential_age += diagnostics.dt;

    diagnostics.projection = canonical_orbit_project_to_eulerian(
        state.atlas, geometry, p, state.eulerian_f);
    WeightedPICDepositedMoments open_deposition;
    WeightedPICDepositedMoments detached_deposition;
    const Moments moments = canonical_boltzmann_poisson_event_moments(
        state,
        geometry,
        radius,
        w,
        eta,
        wq,
        etaq,
        p,
        &open_deposition,
        &detached_deposition);
    diagnostics.potential_variation =
        canonical_mass_weighted_potential_variation_limit(
            moments,
            state.field,
            geometry,
            radius,
            state.time,
            p,
            state.hidden_physical_mass);

    diagnostics.open_mass = canonical_open_marker_mass(state.collision);
    diagnostics.detached_mass =
        canonical_detached_marker_mass(state.collision);
    diagnostics.physical_mass = physical_factor
        * diagnostics.projection.source_mass + diagnostics.open_mass
        + diagnostics.detached_mass;
    diagnostics.open_mass_fraction = state.initial_physical_mass > 0.0
        ? diagnostics.open_mass / state.initial_physical_mass
        : 0.0;
    diagnostics.active_open_mass = open_deposition.active_marker_mass;
    diagnostics.hidden_open_mass = open_deposition.hidden_inner_mass;
    diagnostics.escaped_open_mass = open_deposition.escaped_mass;
    diagnostics.detached_mass_fraction = state.initial_physical_mass > 0.0
        ? diagnostics.detached_mass / state.initial_physical_mass : 0.0;
    diagnostics.active_detached_mass = detached_deposition.active_marker_mass;
    diagnostics.hidden_detached_mass = detached_deposition.hidden_inner_mass;
    diagnostics.escaped_detached_mass = detached_deposition.escaped_mass;
    for (const WeightedPICMarker& marker : state.collision.open_marker) {
        if (marker.state == WeightedPICMarkerState::active) {
            ++diagnostics.active_open_markers;
        } else if (marker.state == WeightedPICMarkerState::delayed_inner) {
            ++diagnostics.hidden_open_markers;
        } else {
            ++diagnostics.escaped_open_markers;
        }
    }
    for (const WeightedPICMarker& marker : state.collision.detached_marker) {
        if (marker.state == WeightedPICMarkerState::active) {
            ++diagnostics.active_detached_markers;
        } else if (marker.state == WeightedPICMarkerState::delayed_inner) {
            ++diagnostics.hidden_detached_markers;
        } else {
            ++diagnostics.escaped_detached_markers;
        }
    }
    state.hidden_physical_mass = physical_factor
        * diagnostics.projection.hidden_mass + diagnostics.hidden_open_mass
        + diagnostics.hidden_detached_mass;
    VlasovPoissonField new_field = solve_vlasov_poisson_field(
        moments, radius, state.time, p, state.hidden_physical_mass);
    const CanonicalFieldMismatch field_mismatch = canonical_field_mismatch(
        state.field,
        new_field,
        moments,
        geometry,
        state.hidden_physical_mass,
        options.potential_mass_fraction_floor);
    diagnostics.relative_potential_mismatch = field_mismatch.potential;
    diagnostics.relative_force_mismatch = field_mismatch.force;
    diagnostics.mass_weighted_potential_mismatch =
        field_mismatch.potential_mass_weighted_rms;
    diagnostics.mass_weighted_force_mismatch =
        field_mismatch.force_mass_weighted_rms;
    const CanonicalFieldMismatch field_change = canonical_field_mismatch(
        state.poisson_reference,
        new_field,
        moments,
        geometry,
        state.hidden_physical_mass,
        options.potential_mass_fraction_floor);
    diagnostics.mass_weighted_potential_change =
        field_change.potential_mass_weighted_rms;
    diagnostics.mass_weighted_force_change =
        field_change.force_mass_weighted_rms;
    const double maximum_field_change = field_change.clock_norm();
    if (state.potential_age > 0.0 && maximum_field_change > 0.0) {
        const double relative_change_rate = maximum_field_change
            / state.potential_age;
        state.potential_timescale = 1.0 / relative_change_rate;
        state.potential_deadline_age = state.potential_age
            + std::max(
                0.0,
                options.potential_fraction - maximum_field_change)
                / relative_change_rate;
    } else {
        state.potential_timescale = std::numeric_limits<double>::infinity();
        state.potential_deadline_age = std::numeric_limits<double>::infinity();
    }
    if (diagnostics.open_mass_fraction
        > options.maximum_open_mass_fraction) {
        throw std::runtime_error(
            "Canonical open-orbit carrier exceeded its configured mass fraction.");
    }
    const double collision_excluded_open_fraction =
        state.initial_physical_mass > 0.0
        ? (diagnostics.hidden_open_mass
            + diagnostics.hidden_detached_mass)
            / state.initial_physical_mass
        : 0.0;
    if (collision_excluded_open_fraction
        > options.maximum_collision_excluded_open_mass_fraction) {
        throw std::runtime_error(
            "Canonical hidden open-orbit carrier exceeded the collision "
            "sampling budget; reduce the inner boundary.");
    }

    if (maximum_field_change < options.potential_fraction) {
        return diagnostics;
    }

    auto new_potential = [&](double r) {
        return sample_vp_field(
            r, radial_coordinate, radius, new_field, p).phi;
    };
    CanonicalAtlasBuildOptions rebuild_options = options.atlas;
    rebuild_options.stationary_distribution = {};
    rebuild_options.energy_partition_weight = {};
    rebuild_options.action_edge_by_eta =
        canonical_orbit_action_edges(state.atlas, p.NL);
    const auto build_begin = std::chrono::steady_clock::now();
    CanonicalAtlasConstructionResult rebuilt =
        build_canonical_bound_orbit_atlas(
            geometry,
            p,
            options.energy_rows,
            new_potential,
            state.hidden_physical_mass,
            rebuild_options);
    diagnostics.atlas_build_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - build_begin).count();
    diagnostics.transfer = canonical_orbit_transfer_preserved_topology(
        state.atlas, rebuilt.atlas);
    state.atlas = std::move(rebuilt.atlas);
    state.collision.source_orbit_cache.clear();
    state.field = std::move(new_field);
    state.potential_age = 0.0;
    diagnostics.potential_rebuilt = true;
    diagnostics.projection = canonical_orbit_project_to_eulerian(
        state.atlas, geometry, p, state.eulerian_f);
    WeightedPICDepositedMoments rebuilt_open_deposition;
    WeightedPICDepositedMoments rebuilt_detached_deposition;
    const Moments rebuilt_moments = canonical_boltzmann_poisson_event_moments(
        state,
        geometry,
        radius,
        w,
        eta,
        wq,
        etaq,
        p,
        &rebuilt_open_deposition,
        &rebuilt_detached_deposition);
    state.hidden_physical_mass = physical_factor
        * diagnostics.projection.hidden_mass
        + rebuilt_open_deposition.hidden_inner_mass
        + rebuilt_detached_deposition.hidden_inner_mass
        + canonical_signed_increment_mass_by_state(
            state.collision, WeightedPICMarkerState::delayed_inner);
    diagnostics.potential_variation = vp_potential_variation_limit(
        rebuilt_moments, state.field, radius, state.time, p);
    const VlasovPoissonField rebuilt_poisson = solve_vlasov_poisson_field(
        rebuilt_moments,
        radius,
        state.time,
        p,
        state.hidden_physical_mass);
    const CanonicalFieldMismatch rebuilt_mismatch = canonical_field_mismatch(
        state.field,
        rebuilt_poisson,
        rebuilt_moments,
        geometry,
        state.hidden_physical_mass,
        options.potential_mass_fraction_floor);
    if (rebuilt_mismatch.clock_norm() > options.startup_field_tolerance) {
        throw std::runtime_error(
            "Canonical field rebuild left a non-self-consistent atlas.");
    }
    state.poisson_reference = rebuilt_poisson;
    state.potential_timescale = std::numeric_limits<double>::infinity();
    state.potential_deadline_age = std::numeric_limits<double>::infinity();
    return diagnostics;
}

// Reconstruct a self-consistent atlas at the current time.  Preserve action-row
// topology and transfer its mass, then redeposit and test the resulting field.
// Optional iterations repeat this field/distribution consistency calculation.
inline void rebuild_canonical_boltzmann_poisson_event_atlas(
    CanonicalBoltzmannPoissonEventState& state,
    const PhaseSpaceCellGeometry& geometry,
    const std::vector<double>& radial_coordinate,
    const std::vector<double>& radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const std::vector<double>& wq,
    const std::vector<double>& etaq,
    const Params& p,
    const CanonicalBoltzmannPoissonEventOptions& options,
    CanonicalBoltzmannPoissonEventDiagnostics& diagnostics) {
    constexpr double physical_factor = 8.0 * PI * PI;
    if (options.absorb_detached_carriers_on_field_rebuild) {
        absorb_canonical_collision_carriers(
            state,
            geometry,
            radial_coordinate,
            radius,
            p,
            options,
            diagnostics);
    }
    diagnostics.projection = canonical_orbit_project_to_eulerian(
        state.atlas, geometry, p, state.eulerian_f);
    WeightedPICDepositedMoments open_deposition;
    WeightedPICDepositedMoments detached_deposition;
    const Moments moments = canonical_boltzmann_poisson_event_moments(
        state,
        geometry,
        radius,
        w,
        eta,
        wq,
        etaq,
        p,
        &open_deposition,
        &detached_deposition);
    state.hidden_physical_mass = physical_factor
        * diagnostics.projection.hidden_mass
        + open_deposition.hidden_inner_mass
        + detached_deposition.hidden_inner_mass
        + canonical_signed_increment_mass_by_state(
            state.collision, WeightedPICMarkerState::delayed_inner);
    VlasovPoissonField candidate_field = solve_vlasov_poisson_field(
        moments, radius, state.time, p, state.hidden_physical_mass);
    const CanonicalFieldMismatch pre_rebuild_mismatch =
        canonical_field_mismatch(
            state.field,
            candidate_field,
            moments,
            geometry,
            state.hidden_physical_mass,
            options.potential_mass_fraction_floor);
    diagnostics.relative_potential_mismatch =
        pre_rebuild_mismatch.potential;
    diagnostics.relative_force_mismatch = pre_rebuild_mismatch.force;
    diagnostics.mass_weighted_potential_mismatch =
        pre_rebuild_mismatch.potential_mass_weighted_rms;
    diagnostics.mass_weighted_force_mismatch =
        pre_rebuild_mismatch.force_mass_weighted_rms;

    CanonicalFieldMismatch final_mismatch;
    bool converged = false;
    for (int iteration = 0;
         iteration <= options.startup_field_iterations;
         ++iteration) {
        auto candidate_potential = [&](double r) {
            return sample_vp_field(
                r, radial_coordinate, radius, candidate_field, p).phi;
        };
        CanonicalAtlasBuildOptions rebuild_options = options.atlas;
        rebuild_options.stationary_distribution = {};
        rebuild_options.energy_partition_weight = {};
        rebuild_options.action_edge_by_eta =
            canonical_orbit_action_edges(state.atlas, p.NL);
        const auto build_begin = std::chrono::steady_clock::now();
        CanonicalAtlasConstructionResult rebuilt =
            build_canonical_bound_orbit_atlas(
                geometry,
                p,
                options.energy_rows,
                candidate_potential,
                state.hidden_physical_mass,
                rebuild_options);
        diagnostics.atlas_build_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - build_begin).count();
        const CanonicalPreservedTopologyTransfer transfer =
            canonical_orbit_transfer_preserved_topology(
                state.atlas, rebuilt.atlas);
        diagnostics.transfer.mass_before += transfer.mass_before;
        diagnostics.transfer.mass_after += transfer.mass_after;
        diagnostics.transfer.residual += transfer.residual;
        diagnostics.transfer.maximum_normalized_action_edge_shift = std::max(
            diagnostics.transfer.maximum_normalized_action_edge_shift,
            transfer.maximum_normalized_action_edge_shift);
        diagnostics.transfer.transferred_rows += transfer.transferred_rows;
        diagnostics.transfer.changed_row_kinds += transfer.changed_row_kinds;
        // A field update changes the orbit coordinates.  The conservative
        // transfer must close before the new atlas replaces the current one.
        const double transfer_scale = std::max(1.0, transfer.mass_before);
        if (std::abs(transfer.residual)
            > 2.0e-12 * transfer_scale) {
            throw std::runtime_error(
                "Canonical field rebuild failed its atlas mass ledger.");
        }
        state.atlas = std::move(rebuilt.atlas);
        state.collision.source_orbit_cache.clear();
        state.field = candidate_field;
        diagnostics.projection = canonical_orbit_project_to_eulerian(
            state.atlas, geometry, p, state.eulerian_f);
        WeightedPICDepositedMoments rebuilt_open;
        WeightedPICDepositedMoments rebuilt_detached;
        const Moments rebuilt_moments =
            canonical_boltzmann_poisson_event_moments(
                state,
                geometry,
                radius,
                w,
                eta,
                wq,
                etaq,
                p,
                &rebuilt_open,
                &rebuilt_detached);
        state.hidden_physical_mass = physical_factor
            * diagnostics.projection.hidden_mass
            + rebuilt_open.hidden_inner_mass
            + rebuilt_detached.hidden_inner_mass
            + canonical_signed_increment_mass_by_state(
                state.collision, WeightedPICMarkerState::delayed_inner);
        const VlasovPoissonField poisson = solve_vlasov_poisson_field(
            rebuilt_moments,
            radius,
            state.time,
            p,
            state.hidden_physical_mass);
        final_mismatch = canonical_field_mismatch(
            state.field,
            poisson,
            rebuilt_moments,
            geometry,
            state.hidden_physical_mass,
            options.potential_mass_fraction_floor);
        // Accept the new frozen potential only after its deposited density
        // reproduces the field within the configured mass-weighted tolerance.
        if (final_mismatch.clock_norm()
            <= options.startup_field_tolerance) {
            state.poisson_reference = poisson;
            converged = true;
            break;
        }
        candidate_field = poisson;
    }
    if (!converged) {
        throw std::runtime_error(
            "Canonical field rebuild left a non-self-consistent atlas; "
            "mass-weighted mismatch="
            + std::to_string(final_mismatch.clock_norm()) + ".");
    }
    diagnostics.potential_rebuilt = true;
}

// Advance to the earlier field or collision deadline.  A zero orbit interval
// can service an already-due event while preserving the physical time.
inline CanonicalBoltzmannPoissonEventDiagnostics
advance_canonical_boltzmann_poisson_event(
    CanonicalBoltzmannPoissonEventState& state,
    const PhaseSpaceCellGeometry& geometry,
    const std::vector<double>& radial_coordinate,
    const std::vector<double>& radius,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const std::vector<double>& wq,
    const std::vector<double>& etaq,
    const Params& p,
    const CanonicalBoltzmannPoissonEventOptions& options,
    double remaining_time = std::numeric_limits<double>::infinity()) {
    validate_canonical_boltzmann_poisson_event_options(options);
    if (!(remaining_time > 0.0) || std::isnan(remaining_time)) {
        throw std::runtime_error(
            "Canonical Boltzmann-Poisson event needs positive remaining time.");
    }

    CanonicalBoltzmannPoissonEventDiagnostics diagnostics;
    // The physical collision estimate and the conservative proposal-rate bound
    // both constrain scheduling; the safety factor acts on the latter.
    double scheduled_collision_timescale = state.collision_timescale;
    if (std::isfinite(state.collision_majorant_timescale)) {
        scheduled_collision_timescale = std::min(
            scheduled_collision_timescale,
            options.collision_cfl_safety
                * state.collision_majorant_timescale
                / options.collision_fraction);
    }
    diagnostics.plan = plan_weighted_pic_event(
        {state.potential_age, state.collision_age},
        state.potential_timescale,
        scheduled_collision_timescale,
        options.potential_fraction,
        options.collision_fraction,
        std::min(options.maximum_collision_interval, remaining_time));
    diagnostics.dt = diagnostics.plan.orbit_interval;
    if (!(diagnostics.dt >= 0.0) || !std::isfinite(diagnostics.dt)) {
        throw std::runtime_error(
            "Canonical event planner selected an invalid interval: dt="
            + std::to_string(diagnostics.dt)
            + ", potential_age=" + std::to_string(state.potential_age)
            + ", collision_age=" + std::to_string(state.collision_age)
            + ", potential_timescale="
            + std::to_string(state.potential_timescale)
            + ", collision_timescale="
            + std::to_string(scheduled_collision_timescale) + ".");
    }

    constexpr double physical_factor = 8.0 * PI * PI;
    // Transport changes phase and boundary state before either due update.
    // The signed carrier sum closes the same represented mass on both sides.
    diagnostics.transport_mass_before = physical_factor
        * canonical_orbit_atlas_mass(state.atlas)
        + canonical_total_carrier_mass(state.collision);
    if (diagnostics.dt > 0.0) {
        const auto open_transport_begin = std::chrono::steady_clock::now();
        advance_canonical_open_orbit_carrier(
            state.collision.open_marker,
            diagnostics.dt,
            radial_coordinate,
            radius,
            state.field,
            p,
            state.hidden_physical_mass,
            options.collision.threads);
        advance_canonical_open_orbit_carrier(
            state.collision.detached_marker,
            diagnostics.dt,
            radial_coordinate,
            radius,
            state.field,
            p,
            state.hidden_physical_mass,
            options.collision.threads);
        advance_canonical_open_orbit_carrier(
            state.collision.signed_increment_positive,
            diagnostics.dt,
            radial_coordinate,
            radius,
            state.field,
            p,
            state.hidden_physical_mass,
            options.collision.threads);
        advance_canonical_open_orbit_carrier(
            state.collision.signed_increment_negative,
            diagnostics.dt,
            radial_coordinate,
            radius,
            state.field,
            p,
            state.hidden_physical_mass,
            options.collision.threads);
        diagnostics.open_transport_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - open_transport_begin).count();
        canonical_orbit_advance_frozen(state.atlas, diagnostics.dt);
    }
    diagnostics.transport_mass_after = physical_factor
        * canonical_orbit_atlas_mass(state.atlas)
        + canonical_total_carrier_mass(state.collision);
    diagnostics.transport_mass_residual =
        diagnostics.transport_mass_after
        - diagnostics.transport_mass_before;
    state.time += diagnostics.dt;
    ++state.step;

    // A collision uses all time accumulated since the previous collision,
    // including the interval just transported in the current frozen field.
    auto collide = [&] {
        const double collision_interval =
            state.collision_age + diagnostics.dt;
        const auto collision_begin = std::chrono::steady_clock::now();
        auto potential = [&](double r) {
            return sample_vp_field(
                r, radial_coordinate, radius, state.field, p).phi;
        };
        // Materialize the previous signed increment before resampling.
        // Active holes retained by that operation require an admissible atlas
        // lift before a new collision evaluation can consume the distribution.
        if (options.collision_coupling
            == CanonicalCollisionCoupling::delta_ntc) {
            diagnostics.signed_increment_materialization =
                canonical_materialize_signed_collision_increment(
                    state.atlas,
                    geometry,
                    p,
                    state.hidden_physical_mass,
                    options.collision
                        .maximum_lift_phase_interpolation_cell_error,
                    options.collision.threads,
                    options.collision.open_compression,
                    state.collision,
                    potential);
            const auto& retained =
                diagnostics.signed_increment_materialization;
            const std::size_t retained_active = std::accumulate(
                retained.retained_open_markers_by_reason.begin(),
                retained.retained_open_markers_by_reason.end(),
                std::size_t{0});
            if (retained_active > 0) {
                std::ostringstream message;
                message.precision(17);
                message
                    << "Canonical signed collision increment has "
                    << retained_active
                    << " active packets outside the current atlas lift: "
                    << "eta="
                    << retained.retained_open_markers_by_reason[
                        static_cast<std::size_t>(
                            CanonicalNTCOpenReason::eta_outside_atlas)]
                    << ", missing_eta_row="
                    << retained.retained_open_markers_by_reason[
                        static_cast<std::size_t>(
                            CanonicalNTCOpenReason::eta_row_missing)]
                    << ", high_energy="
                    << retained.retained_open_markers_by_reason[
                        static_cast<std::size_t>(
                            CanonicalNTCOpenReason::energy_above_atlas)]
                    << ", energy_gap="
                    << retained.retained_open_markers_by_reason[
                        static_cast<std::size_t>(
                            CanonicalNTCOpenReason::energy_outside_row)]
                    << ", orbit_geometry="
                    << retained.retained_open_markers_by_reason[
                        static_cast<std::size_t>(
                            CanonicalNTCOpenReason::orbit_geometry_failure)];
                throw std::runtime_error(message.str());
            }
        }
        diagnostics.collision_represented_mass_before = physical_factor
            * canonical_orbit_atlas_mass(state.atlas)
            + canonical_total_carrier_mass(state.collision);
        if (options.collision_coupling
            == CanonicalCollisionCoupling::reversible_increment) {
            validate_canonical_reversible_collision_carriers(state, options);
            std::vector<WeightedPICMarker> open_increment;
            diagnostics.reversible_collision =
                apply_canonical_reversible_collision_increment(
                    state.atlas,
                    geometry,
                    radius,
                    w,
                    eta,
                    wq,
                    etaq,
                    collision_interval,
                    p,
                    options.maximum_collision_escaped_increment_fraction,
                    options.maximum_collision_moment_repair_mass_fraction,
                    &open_increment,
                    &state.eulerian_f,
                    &state.streamed_collision_workspace);
            canonical_append_open_collision_increment(
                state.collision, std::move(open_increment));
            // Positivity of the atlas increment and the user interval cap add
            // independent restrictions to the microscopic collision time.
            const double atlas_collision_clock =
                diagnostics.reversible_collision
                    .atlas_mass_weighted_positivity_timescale
                / options.collision_fraction;
            const double maximum_interval_clock =
                options.maximum_collision_interval
                / options.collision_fraction;
            state.collision_timescale = std::min(
                {
                    diagnostics.reversible_collision.collision
                        .collision_time_estimate,
                    atlas_collision_clock,
                    maximum_interval_clock,
                });
            state.collision_majorant_timescale =
                std::numeric_limits<double>::infinity();
        } else {
            CanonicalNTCCollisionOptions collision_options =
                options.collision;
            collision_options.hidden_central_mass =
                state.hidden_physical_mass;
            collision_options.retain_collision_products_as_detached_carriers =
                options.collision_coupling
                    != CanonicalCollisionCoupling::owned_ntc;
            collision_options.retain_collision_increment_as_signed_carriers =
                options.collision_coupling
                    == CanonicalCollisionCoupling::delta_ntc;
            if (collision_options
                    .retain_collision_increment_as_signed_carriers) {
                collision_options
                    .retain_collision_products_as_detached_carriers = false;
            }
            if (options.collision_coupling
                == CanonicalCollisionCoupling::owned_ntc) {
                diagnostics.collision = apply_canonical_owned_ntc_collision(
                    state.atlas,
                    geometry,
                    collision_interval,
                    p,
                    collision_options,
                    state.collision,
                    potential);
            } else {
                diagnostics.collision = apply_canonical_ntc_collision(
                    state.atlas,
                    geometry,
                    collision_interval,
                    p,
                    collision_options,
                    state.collision,
                    potential);
            }
            state.collision_timescale =
                diagnostics.collision.physical_collision_timescale;
            state.collision_majorant_timescale =
                diagnostics.collision.collision_majorant_timescale;
        }
        diagnostics.collision_represented_mass_after = physical_factor
            * canonical_orbit_atlas_mass(state.atlas)
            + canonical_total_carrier_mass(state.collision);
        diagnostics.collision_represented_mass_residual =
            diagnostics.collision_represented_mass_after
            - diagnostics.collision_represented_mass_before;
        const double collision_mass_scale = std::max(
            1.0, diagnostics.collision_represented_mass_before);
        if (std::abs(diagnostics.collision_represented_mass_residual)
            > 2.0e-11 * collision_mass_scale) {
            std::ostringstream message;
            message.precision(17);
            message
                << "Canonical collision failed its represented total-mass ledger"
                << ": before="
                << diagnostics.collision_represented_mass_before
                << ", after="
                << diagnostics.collision_represented_mass_after
                << ", residual="
                << diagnostics.collision_represented_mass_residual
                << ", atlas_residual_reduced="
                << diagnostics.reversible_collision.atlas_mass_residual
                << ", escaped_increment_reduced="
                << diagnostics.reversible_collision.escaped_increment_mass
                << ", escaped_positive_physical="
                << physical_factor
                    * diagnostics.reversible_collision
                        .escaped_positive_increment_mass
                << ".";
            throw std::runtime_error(message.str());
        }
        diagnostics.collision_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - collision_begin).count();
        diagnostics.collision_applied = true;
    };
    auto rebuild = [&] {
        rebuild_canonical_boltzmann_poisson_event_atlas(
            state,
            geometry,
            radial_coordinate,
            radius,
            w,
            eta,
            wq,
            etaq,
            p,
            options,
            diagnostics);
    };

    // When both deadlines coincide, retain the planner's specified ordering.
    // The second operation therefore sees the state produced by the first.
    if (diagnostics.plan.potential_due
        && diagnostics.plan.collision_due) {
        if (diagnostics.plan.order
            == WeightedPICEventOrder::reconstruction_then_collision) {
            rebuild();
            collide();
        } else {
            collide();
            rebuild();
        }
    } else if (diagnostics.plan.potential_due) {
        rebuild();
    } else if (diagnostics.plan.collision_due) {
        collide();
    }

    const bool carrier_budget_exceeded =
        options.absorb_detached_carriers_on_field_rebuild
        &&
        canonical_collision_retains_exact_carriers(
            options.collision_coupling)
        && state.initial_physical_mass > 0.0
        && canonical_active_detached_marker_mass(state.collision)
            > options.maximum_detached_mass_fraction
                * state.initial_physical_mass;
    if (carrier_budget_exceeded && !diagnostics.potential_rebuilt) {
        // Carrier products remain at their exact collision positions between
        // field events.  Reaching the active detached-mass budget advances
        // the next field event and its associated atlas reconstruction.
        rebuild();
        diagnostics.plan.potential_due = true;
        diagnostics.plan.order = diagnostics.collision_applied
            ? WeightedPICEventOrder::collision_then_reconstruction
            : WeightedPICEventOrder::reconstruction_only;
    }

    // Reset only the ages of updates actually performed; the other clock keeps
    // its accumulated physical exposure across this event.
    WeightedPICEventClockState clock{
        state.potential_age, state.collision_age};
    commit_weighted_pic_event(clock, diagnostics.plan);
    state.potential_age = clock.potential_age;
    state.collision_age = clock.collision_age;

    diagnostics.projection = canonical_orbit_project_to_eulerian(
        state.atlas, geometry, p, state.eulerian_f);
    WeightedPICDepositedMoments open_deposition;
    WeightedPICDepositedMoments detached_deposition;
    const Moments moments = canonical_boltzmann_poisson_event_moments(
        state,
        geometry,
        radius,
        w,
        eta,
        wq,
        etaq,
        p,
        &open_deposition,
        &detached_deposition);
    diagnostics.open_mass = canonical_open_marker_mass(state.collision);
    diagnostics.detached_mass =
        canonical_detached_marker_mass(state.collision);
    diagnostics.physical_mass = physical_factor
        * diagnostics.projection.source_mass
        + canonical_total_carrier_mass(state.collision);
    diagnostics.open_mass_fraction = state.initial_physical_mass > 0.0
        ? diagnostics.open_mass / state.initial_physical_mass
        : 0.0;
    diagnostics.active_open_mass = open_deposition.active_marker_mass;
    diagnostics.hidden_open_mass = open_deposition.hidden_inner_mass;
    diagnostics.escaped_open_mass = open_deposition.escaped_mass;
    diagnostics.detached_mass_fraction = state.initial_physical_mass > 0.0
        ? diagnostics.detached_mass / state.initial_physical_mass : 0.0;
    diagnostics.active_detached_mass = detached_deposition.active_marker_mass;
    diagnostics.hidden_detached_mass = detached_deposition.hidden_inner_mass;
    diagnostics.escaped_detached_mass = detached_deposition.escaped_mass;
    for (const WeightedPICMarker& marker : state.collision.open_marker) {
        if (marker.state == WeightedPICMarkerState::active) {
            ++diagnostics.active_open_markers;
        } else if (marker.state == WeightedPICMarkerState::delayed_inner) {
            ++diagnostics.hidden_open_markers;
        } else {
            ++diagnostics.escaped_open_markers;
        }
    }
    for (const WeightedPICMarker& marker : state.collision.detached_marker) {
        if (marker.state == WeightedPICMarkerState::active) {
            ++diagnostics.active_detached_markers;
        } else if (marker.state == WeightedPICMarkerState::delayed_inner) {
            ++diagnostics.hidden_detached_markers;
        } else {
            ++diagnostics.escaped_detached_markers;
        }
    }
    const double canonical_hidden_mass = physical_factor
        * diagnostics.projection.hidden_mass;
    state.hidden_physical_mass = canonical_hidden_mass
        + diagnostics.hidden_open_mass
        + diagnostics.hidden_detached_mass
        + canonical_signed_increment_mass_by_state(
            state.collision, WeightedPICMarkerState::delayed_inner);

    // Reconstruct a diagnostic field from all current mass, even while the
    // transport field stays frozen.  Its change sets the next field deadline.
    const VlasovPoissonField current_poisson = solve_vlasov_poisson_field(
        moments, radius, state.time, p, state.hidden_physical_mass);
    const CanonicalFieldMismatch field_mismatch = canonical_field_mismatch(
        state.field,
        current_poisson,
        moments,
        geometry,
        state.hidden_physical_mass,
        options.potential_mass_fraction_floor);
    diagnostics.relative_potential_mismatch = field_mismatch.potential;
    diagnostics.relative_force_mismatch = field_mismatch.force;
    diagnostics.mass_weighted_potential_mismatch =
        field_mismatch.potential_mass_weighted_rms;
    diagnostics.mass_weighted_force_mismatch =
        field_mismatch.force_mass_weighted_rms;
    const CanonicalFieldMismatch field_change = canonical_field_mismatch(
        state.poisson_reference,
        current_poisson,
        moments,
        geometry,
        state.hidden_physical_mass,
        options.potential_mass_fraction_floor);
    diagnostics.mass_weighted_potential_change =
        field_change.potential_mass_weighted_rms;
    diagnostics.mass_weighted_force_change =
        field_change.force_mass_weighted_rms;
    const double maximum_field_change = field_change.clock_norm();
    diagnostics.potential_variation =
        canonical_mass_weighted_potential_variation_limit(
            moments,
            state.field,
            geometry,
            radius,
            state.time,
            p,
            state.hidden_physical_mass);
    // Combine the continuity estimate with the measured change since the last
    // field reference, using the smaller resolved field-variation timescale.
    const double elapsed_field_time = diagnostics.plan.potential_due
        ? diagnostics.dt : state.potential_age;
    const double measured_timescale = elapsed_field_time > 0.0
        && maximum_field_change > 0.0
        ? elapsed_field_time / maximum_field_change
        : std::numeric_limits<double>::infinity();
    state.potential_timescale = std::min(
        diagnostics.potential_variation.timescale,
        measured_timescale);
    state.potential_deadline_age = std::isfinite(state.potential_timescale)
        ? options.potential_fraction * state.potential_timescale
        : std::numeric_limits<double>::infinity();

    if (diagnostics.open_mass_fraction
        > options.maximum_open_mass_fraction) {
        throw std::runtime_error(
            "Canonical open-orbit carrier exceeded its configured mass fraction.");
    }
    // Hidden inner mass contributes to gravity while it lies outside the
    // resolved collision sampler; enforce its physical mass budget explicitly.
    const double collision_excluded_mass = canonical_hidden_mass
        + diagnostics.hidden_open_mass
        + diagnostics.hidden_detached_mass
        + canonical_signed_increment_mass_by_state(
            state.collision, WeightedPICMarkerState::delayed_inner);
    const double collision_excluded_fraction = state.initial_physical_mass > 0.0
        ? collision_excluded_mass / state.initial_physical_mass
        : 0.0;
    if (collision_excluded_fraction
        > options.maximum_collision_excluded_open_mass_fraction) {
        throw std::runtime_error(
            "Canonical hidden carrier exceeded the collision sampling budget; "
            "reduce the inner boundary.");
    }
    return diagnostics;
}

}  // namespace ev
