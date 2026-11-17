// Tests nonnegative joint marker-pair projection and exact mass, radial
// momentum, kinetic-energy, and continuous-carrier budgets.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "ev/weighted_vlasov_pic_phase_remap.hpp"
#include "ev/weighted_vlasov_pic_collisions.hpp"

namespace {

ev::Params remap_params() {
    ev::Params p;
    p.physics_model = "vlasov_poisson";
    p.vp_universe = "static";
    p.vp_interpolation = "hamiltonian_sldg";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "uniform";
    p.Nr = 3;
    p.Nw = 24;
    p.NL = 64;
    p.r_min = 0.5;
    p.r_max = 2.0;
    p.radial_grid = "linear";
    p.w_max = 6.0;
    p.L_max = 8.0;
    p.bp_reversible_candidate_radius = 4;
    p.bp_reversible_projection_tolerance = 2.0e-10;
    return p;
}

bool close(double first, double second, double tolerance = 2.0e-10) {
    return std::abs(first - second)
        <= tolerance * std::max({1.0, std::abs(first), std::abs(second)});
}

void test_pair_projection_and_state_budgets() {
    const ev::Params p = remap_params();
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    constexpr double marker_mass = 0.0125;
    std::vector<ev::WeightedPICMarker> markers;
    for (int i = 0; i < 39; ++i) {
        const int radial_cell = i / 13;
        const int velocity_cell = (7 * i + 5) % p.Nw;
        const int eta_cell = (5 * i + 3) % p.NL;
        ev::WeightedPICMarker marker;
        marker.mass = marker_mass;
        marker.r = geometry.radial.center[static_cast<std::size_t>(radial_cell)]
            + 0.15 * geometry.radial.width[static_cast<std::size_t>(radial_cell)]
                * std::sin(0.43 * i);
        marker.w = geometry.radial_velocity.center[
            static_cast<std::size_t>(velocity_cell)];
        marker.eta = geometry.eta.center[static_cast<std::size_t>(eta_cell)];
        marker.id = static_cast<std::uint64_t>(1000 + i);
        markers.push_back(marker);
    }
    ev::WeightedPICMarker hidden = markers.front();
    hidden.id = 9001;
    hidden.state = ev::WeightedPICMarkerState::delayed_inner;
    hidden.remaining_inner_time = 0.2;
    markers.push_back(hidden);
    ev::WeightedPICMarker escaped = markers.front();
    escaped.id = 9002;
    escaped.state = ev::WeightedPICMarkerState::escaped;
    markers.push_back(escaped);

    const ev::WeightedPICPhaseRemapResult remap =
        ev::remap_weighted_vlasov_pic_to_phase_space(
            markers, geometry, p);
    const double deposited_mass = std::accumulate(
        remap.cell_mass.begin(), remap.cell_mass.end(), 0.0);
    assert(remap.diagnostics.active_markers == 39);
    assert(remap.diagnostics.projected_pairs == 21);
    assert(remap.diagnostics.duplicated_singletons >= 1);
    assert(close(deposited_mass, 39.0 * marker_mass, 3.0e-13));
    assert(close(remap.diagnostics.hidden_inner_mass, marker_mass, 3.0e-13));
    assert(close(remap.diagnostics.escaped_mass, marker_mass, 3.0e-13));
    assert(close(
        remap.diagnostics.radial_momentum_after,
        remap.diagnostics.radial_momentum_before));
    assert(close(
        remap.diagnostics.kinetic_energy_after,
        remap.diagnostics.kinetic_energy_before));
    assert(remap.diagnostics.maximum_projection_error < 2.0e-8);
    assert(std::all_of(
        remap.f.begin(), remap.f.end(), [](double value) {
            return value >= 0.0 && std::isfinite(value);
        }));
    const ev::KineticInvariantLedger ledger =
        ev::evaluate_kinetic_invariant_ledger(remap.f, geometry, p);
    assert(close(ledger.mass, deposited_mass, 3.0e-13));
    assert(close(
        ledger.radial_momentum,
        remap.diagnostics.radial_momentum_before));
    assert(close(
        ledger.kinetic_energy,
        remap.diagnostics.kinetic_energy_before));
    std::cout << "weighted PIC phase remap: active_mass="
              << deposited_mass
              << " momentum_error="
              << remap.diagnostics.radial_momentum_after
                    - remap.diagnostics.radial_momentum_before
              << " energy_error="
              << remap.diagnostics.kinetic_energy_after
                    - remap.diagnostics.kinetic_energy_before
              << '\n';
}

void test_unequal_mass_groups_preserve_all_remap_invariants() {
    const ev::Params p = remap_params();
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    std::vector<ev::WeightedPICMarker> markers;
    long double expected_mass = 0.0L;
    for (int i = 0; i < p.Nr; ++i) {
        // Five distinct weights exercise even groups, odd groups, and unique
        // singleton weights without changing any physical marker.
        for (int q = 0; q < 23; ++q) {
            const int j = 5 + (3 * q + i) % 12;
            const int k = 3 + (7 * q + 2 * i) % 20;
            ev::WeightedPICMarker marker;
            marker.mass = q == 21
                ? 0.017
                : (q == 22
                       ? std::nextafter(
                           0.017,
                           std::numeric_limits<double>::infinity())
                       : 0.0025 * static_cast<double>(1 + q % 5));
            marker.r = geometry.radial.center[static_cast<std::size_t>(i)];
            marker.w = geometry.radial_velocity.center[
                static_cast<std::size_t>(j)];
            marker.eta = geometry.eta.center[static_cast<std::size_t>(k)];
            marker.id = static_cast<std::uint64_t>(markers.size());
            expected_mass += static_cast<long double>(marker.mass);
            markers.push_back(marker);
        }
    }

    const ev::WeightedPICPhaseRemapResult remap =
        ev::remap_weighted_vlasov_pic_to_phase_space(
            markers, geometry, p);
    const double deposited_mass = std::accumulate(
        remap.cell_mass.begin(), remap.cell_mass.end(), 0.0);
    assert(remap.diagnostics.active_markers == markers.size());
    assert(remap.diagnostics.duplicated_singletons > 0);
    assert(close(
        deposited_mass,
        static_cast<double>(expected_mass),
        3.0e-13));
    assert(close(
        remap.diagnostics.radial_momentum_after,
        remap.diagnostics.radial_momentum_before));
    assert(close(
        remap.diagnostics.kinetic_energy_after,
        remap.diagnostics.kinetic_energy_before));
    assert(std::all_of(
        remap.f.begin(), remap.f.end(), [](double value) {
            return value >= 0.0 && std::isfinite(value);
        }));
    const ev::KineticInvariantLedger ledger =
        ev::evaluate_kinetic_invariant_ledger(remap.f, geometry, p);
    assert(close(ledger.mass, deposited_mass, 3.0e-13));
    assert(close(
        ledger.radial_momentum,
        remap.diagnostics.radial_momentum_before));
    assert(close(
        ledger.kinetic_energy,
        remap.diagnostics.kinetic_energy_before));
}

void test_out_of_domain_velocity_fails() {
    const ev::Params p = remap_params();
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    std::vector<ev::WeightedPICMarker> markers(2);
    for (std::size_t i = 0; i < markers.size(); ++i) {
        markers[i].mass = 0.5;
        markers[i].r = 1.0;
        markers[i].w = i == 0 ? 20.0 : -20.0;
        markers[i].eta = 0.1;
        markers[i].id = i;
    }
    bool threw = false;
    try {
        ev::WeightedPICPhaseRemapOptions options;
        options.maximum_carrier_mass_fraction = 0.9;
        (void)ev::remap_weighted_vlasov_pic_to_phase_space(
            markers, geometry, p, options);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

void test_in_domain_convex_hull_carrier_closes_budgets() {
    const ev::Params p = remap_params();
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    constexpr double marker_mass = 0.25;
    constexpr double carrier_marker_mass = 0.125;
    std::vector<ev::WeightedPICMarker> markers(3);

    // This marker is inside the velocity-domain edges.  Its zero kinetic
    // energy lies below every positive cell-centre energy, so no nonnegative
    // combination of cell-centre packets can represent it exactly.
    markers[0].mass = carrier_marker_mass;
    markers[0].r = geometry.radial.center[0];
    markers[0].w = 0.0;
    markers[0].eta = 0.0;
    markers[0].id = 10;

    // The second radial cell contains an exactly representable pair.
    for (std::size_t q = 1; q < markers.size(); ++q) {
        markers[q].mass = marker_mass;
        markers[q].r = geometry.radial.center[1];
        markers[q].w = geometry.radial_velocity.center[10 + q];
        markers[q].eta = geometry.eta.center[4 + q];
        markers[q].id = 10 + q;
    }

    bool strict_threw = false;
    try {
        (void)ev::remap_weighted_vlasov_pic_to_phase_space(
            markers, geometry, p);
    } catch (const std::runtime_error&) {
        strict_threw = true;
    }
    assert(strict_threw);

    ev::WeightedPICPhaseRemapOptions options;
    options.maximum_candidate_radius = 8;
    options.maximum_carrier_mass_fraction = 0.5;
    const ev::WeightedPICPhaseRemapResult remap =
        ev::remap_weighted_vlasov_pic_to_phase_space(
            markers, geometry, p, options);
    const ev::KineticInvariantLedger grid_ledger =
        ev::evaluate_kinetic_invariant_ledger(remap.f, geometry, p);

    assert(remap.diagnostics.carrier_pairs == 1);
    assert(remap.diagnostics.carrier_markers == 1);
    assert(remap.carrier_marker_indices.size() == 1);
    assert(remap.carrier_marker_indices.front() == 0);
    assert(close(
        remap.diagnostics.carrier_mass,
        carrier_marker_mass,
        3.0e-13));
    assert(close(
        grid_ledger.mass + remap.diagnostics.carrier_mass,
        remap.diagnostics.active_mass,
        3.0e-13));
    assert(close(
        grid_ledger.radial_momentum
            + remap.diagnostics.carrier_radial_momentum,
        remap.diagnostics.radial_momentum_before));
    assert(close(
        grid_ledger.kinetic_energy
            + remap.diagnostics.carrier_kinetic_energy,
        remap.diagnostics.kinetic_energy_before));
    assert(close(
        remap.diagnostics.radial_momentum_after,
        remap.diagnostics.radial_momentum_before));
    assert(close(
        remap.diagnostics.kinetic_energy_after,
        remap.diagnostics.kinetic_energy_before));
}

void test_collide_then_remap() {
    ev::Params p = remap_params();
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 0.35;
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    constexpr std::size_t per_shell = 1024;
    constexpr double marker_mass = 1.0e-3;
    std::vector<ev::WeightedPICMarker> markers;
    markers.reserve(per_shell * static_cast<std::size_t>(p.Nr));
    for (int i = 0; i < p.Nr; ++i) {
        for (std::size_t q = 0; q < per_shell; ++q) {
            const int j = 7 + static_cast<int>((5 * q + 3 * i) % 10);
            const int k = 2 + static_cast<int>((7 * q + i) % 7);
            ev::WeightedPICMarker marker;
            marker.mass = marker_mass;
            marker.r = geometry.radial.center[static_cast<std::size_t>(i)]
                + 0.2 * geometry.radial.width[static_cast<std::size_t>(i)]
                    * ((static_cast<double>(q) + 0.5)
                       / static_cast<double>(per_shell) - 0.5);
            marker.w = geometry.radial_velocity.center[static_cast<std::size_t>(j)];
            marker.eta = geometry.eta.center[static_cast<std::size_t>(k)];
            marker.id = markers.size();
            markers.push_back(marker);
        }
    }
    ev::WeightedPICCollisionOptions options;
    options.target_shell_markers = 256;
    options.minimum_shell_markers = 16;
    options.collision_cfl = 1.0;
    ev::WeightedPICCollisionSamplerState sampler{UINT64_C(0x1234abcd), 0};
    ev::WeightedPICCollisionWorkspace workspace;
    const ev::WeightedPICCollisionDiagnostics collision =
        ev::apply_weighted_pic_collision_interval(
            markers,
            ev::make_weighted_pic_radial_mesh(geometry),
            0.08,
            p,
            options,
            sampler,
            workspace,
            2);
    assert(collision.accepted_pairs > 20);
    const ev::WeightedPICPhaseRemapResult remap =
        ev::remap_weighted_vlasov_pic_to_phase_space(
            markers, geometry, p);
    // Analytic equal-mass reference is independent of either reduction.
    // The original binary64 active-mass sum lost 2.27e-13 for these 3072
    // markers and falsely tripped the unchanged production budget guard.
    const double expected_mass = static_cast<double>(markers.size()) * marker_mass;
    assert(close(remap.diagnostics.active_mass, expected_mass,
                 2.0 * std::numeric_limits<double>::epsilon()));
    double grid_mass = 0.0;
    double grid_correction = 0.0;
    for (double mass : remap.cell_mass) {
        const double updated = grid_mass + mass;
        grid_correction += std::abs(grid_mass) >= std::abs(mass)
            ? (grid_mass - updated) + mass
            : (mass - updated) + grid_mass;
        grid_mass = updated;
    }
    grid_mass += grid_correction;
    assert(close(grid_mass, expected_mass,
                 32.0 * std::numeric_limits<double>::epsilon()));
    assert(close(
        remap.diagnostics.radial_momentum_after,
        remap.diagnostics.radial_momentum_before));
    assert(close(
        remap.diagnostics.kinetic_energy_after,
        remap.diagnostics.kinetic_energy_before));
    std::cout << "collide-then-remap: accepted="
              << collision.accepted_pairs
              << " active_mass_error=" << remap.diagnostics.active_mass - expected_mass
              << " grid_mass_error=" << grid_mass - expected_mass
              << " projection_error="
              << remap.diagnostics.maximum_projection_error << '\n';
}

}  // namespace

int main() {
    test_pair_projection_and_state_budgets();
    test_unequal_mass_groups_preserve_all_remap_invariants();
    test_out_of_domain_velocity_fails();
    test_in_domain_convex_hull_carrier_closes_budgets();
    test_collide_then_remap();
    std::cout << "weighted Vlasov PIC phase-remap tests passed\n";
    return 0;
}
