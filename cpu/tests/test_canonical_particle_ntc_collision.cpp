// Tests source-owned canonical DSMC parcels through repeated NTC events,
// one-time withdrawal, final lift, and pair/global invariant closure.

#include "ev/canonical_particle_ntc_collision.hpp"

#include <cassert>
#include <iostream>

namespace {

struct NullResult {
    double relative_l1 = 0.0;
    double event_mass_fraction = 0.0;
    double mass_residual = 0.0;
    double momentum_residual = 0.0;
    double kinetic_residual = 0.0;
    double energy_residual = 0.0;
    std::uint64_t accepted = 0;
    std::vector<double> change;
};

NullResult run_maxwellian_null(
    const ev::CanonicalOrbitAtlas& initial,
    const ev::PhaseSpaceCellGeometry& geometry,
    const ev::Params& p,
    std::size_t marker_budget,
    std::uint64_t seed) {
    ev::CanonicalOrbitAtlas atlas = initial;
    std::vector<double> before;
    ev::canonical_orbit_project_to_eulerian(atlas, geometry, p, before);

    ev::CanonicalNTCCollisionOptions options;
    options.marker_budget = marker_budget;
    options.markers_per_occupied_cell = 16;
    options.threads = 4;
    options.virtual_collision_batches = 128;
    options.maximum_unrepresented_mass_fraction = 5.0e-6;
    options.maximum_lift_phase_interpolation_cell_error = 0.05;
    options.collision.target_shell_markers = 256;
    options.collision.minimum_shell_markers = 16;
    options.collision.collision_cfl = 1.0;
    options.collision.fail_on_nonlocal_shell = false;

    ev::CanonicalNTCCollisionState state;
    state.marker_sampler.seed = seed;
    state.collision_sampler.seed = seed ^ UINT64_C(0x9e3779b97f4a7c15);
    const auto diagnostics = ev::apply_canonical_owned_ntc_collision(
        atlas,
        geometry,
        5.0e-6,
        p,
        options,
        state,
        [](double radius) { return 0.5 * radius * radius; });

    std::vector<double> after;
    ev::canonical_orbit_project_to_eulerian(atlas, geometry, p, after);
    long double difference = 0.0L;
    long double reference = 0.0L;
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t cell = ev::idx(i, j, k, p);
                const double measure = ev::canonical_cell_measure(
                    geometry, i, j, k);
                difference += measure * std::abs(after[cell] - before[cell]);
                reference += measure * std::abs(before[cell]);
            }
        }
    }

    NullResult result;
    result.change.resize(after.size());
    for (std::size_t q = 0; q < after.size(); ++q) {
        result.change[q] = after[q] - before[q];
    }
    result.relative_l1 = static_cast<double>(difference / reference);
    result.event_mass_fraction = diagnostics.collision.represented_event_mass
        / diagnostics.physical_mass_before;
    result.mass_residual = diagnostics.physical_mass_residual;
    result.momentum_residual = diagnostics.physical_radial_momentum_residual;
    result.kinetic_residual = diagnostics.physical_kinetic_energy_residual;
    result.energy_residual = diagnostics.atlas_energy_quantization_error;
    result.accepted = diagnostics.collision.accepted_pairs;
    return result;
}

}  // namespace

int main() {
    ev::Params p;
    p.physics_model = "boltzmann_poisson";
    p.vp_universe = "static";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "power";
    p.kinetic_eta_power = 2.0;
    p.radial_grid = "log";
    p.inner_boundary = "delayed_return";
    p.r_min = 0.01;
    p.r_max = 4.0;
    p.w_max = 4.0;
    p.L_max = 6.0;
    p.Nr = 16;
    p.Nw = 40;
    p.NL = 24;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 2.0;
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);

    ev::CanonicalAtlasBuildOptions atlas_options;
    atlas_options.phase_cells = 12;
    atlas_options.mapped_boundary_segments = 16;
    atlas_options.maximum_patch_depth = 6;
    atlas_options.mapped_volume_tolerance = 2.0e-2;
    atlas_options.maximum_energy = 5.0;
    atlas_options.eta_quadrature_points = 2;
    constexpr double temperature = 0.2;
    atlas_options.stationary_distribution = [](double, double h) {
        return 100.0 * std::exp(-h / temperature);
    };
    atlas_options.energy_partition_weight = [](double, double h) {
        return std::exp(-0.5 * h / temperature);
    };
    auto construction = ev::build_canonical_bound_orbit_atlas(
        geometry,
        p,
        6,
        [](double radius) { return 0.5 * radius * radius; },
        0.0,
        atlas_options);
    assert(construction.diagnostics.eta_rows_built > 0);

    ev::CanonicalOrbitAtlas initial = std::move(construction.atlas);
    assert(ev::canonical_orbit_atlas_mass(initial) > 0.0);

    const NullResult coarse = run_maxwellian_null(
        initial, geometry, p, 32768, UINT64_C(0x123456789abcdef0));
    const NullResult fine = run_maxwellian_null(
        initial, geometry, p, 65536, UINT64_C(0x123456789abcdef0));
    std::vector<double> ensemble_change = coarse.change;
    constexpr int ensemble_members = 4;
    for (int member = 1; member < ensemble_members; ++member) {
        const NullResult sample = run_maxwellian_null(
            initial,
            geometry,
            p,
            32768,
            UINT64_C(0x123456789abcdef0)
                + static_cast<std::uint64_t>(member)
                    * UINT64_C(0x9e3779b97f4a7c15));
        for (std::size_t q = 0; q < ensemble_change.size(); ++q) {
            ensemble_change[q] += sample.change[q];
        }
    }
    std::vector<double> initial_eulerian;
    ev::canonical_orbit_project_to_eulerian(
        initial, geometry, p, initial_eulerian);
    long double ensemble_difference = 0.0L;
    long double ensemble_reference = 0.0L;
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t cell = ev::idx(i, j, k, p);
                const double measure = ev::canonical_cell_measure(
                    geometry, i, j, k);
                ensemble_difference += measure * std::abs(
                    ensemble_change[cell] / ensemble_members);
                ensemble_reference += measure
                    * std::abs(initial_eulerian[cell]);
            }
        }
    }
    const double ensemble_relative_l1 = static_cast<double>(
        ensemble_difference / ensemble_reference);
    std::cout << "Canonical particle NTC Maxwellian: coarse_l1="
              << coarse.relative_l1
              << " fine_l1=" << fine.relative_l1
              << " coarse_events=" << coarse.accepted
              << " fine_events=" << fine.accepted
              << " fine_event_mass_fraction=" << fine.event_mass_fraction
              << " dM=" << fine.mass_residual
              << " dPr=" << fine.momentum_residual
              << " dK=" << fine.kinetic_residual
              << " dH=" << fine.energy_residual
              << " ensemble_l1=" << ensemble_relative_l1 << '\n';

    assert(coarse.accepted > 0);
    assert(fine.accepted > coarse.accepted);
    assert(std::abs(coarse.mass_residual) < 1.0e-12);
    assert(std::abs(fine.mass_residual) < 1.0e-12);
    assert(std::abs(coarse.momentum_residual) < 1.0e-10);
    assert(std::abs(fine.momentum_residual) < 1.0e-10);
    assert(std::abs(coarse.kinetic_residual) < 1.0e-10);
    assert(std::abs(fine.kinetic_residual) < 1.0e-10);
    assert(std::abs(coarse.energy_residual) < 1.0e-9);
    assert(std::abs(fine.energy_residual) < 1.0e-9);
    assert(fine.relative_l1 < 1.0e-2);
    assert(fine.relative_l1 < 1.25 * coarse.relative_l1);
    // This bridge is retained as a diagnostic for continuous NTC lifting.
    // The production grid operator has a separate exact discrete-Maxwellian
    // null test in test_fully_kinetic_modules.
    assert(ensemble_relative_l1 < 1.0e-2);
    return 0;
}
