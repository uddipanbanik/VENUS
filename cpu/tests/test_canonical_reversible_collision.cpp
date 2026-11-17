// Tests increment-only reversible collision coupling to the canonical atlas:
// projection conservation, Maxwellian null state, and entropy sign.

#include "ev/canonical_reversible_collision.hpp"

#include <cassert>
#include <iostream>

int main() {
    {
        ev::CanonicalOrbitRow row;
        row.energy_left = -1.0;
        row.energy_right = 1.0;
        row.phase_mass = {1.0e-14, 1.0, 1.0};
        row.phase_energy_mass = {1.1e-14, -0.2, 0.2};
        row.phase_energy2_mass = {1.21e-14, 0.3, 0.3};
        const long double energy_before = std::accumulate(
            row.phase_energy_mass.begin(),
            row.phase_energy_mass.end(),
            0.0L);
        const long double energy2_before = std::accumulate(
            row.phase_energy2_mass.begin(),
            row.phase_energy2_mass.end(),
            0.0L);
        ev::CanonicalReversibleCollisionDiagnostics repair;
        ev::canonical_repair_row_energy_moments(row, 2.0e-14, repair);
        assert(repair.moment_repaired_mass == 1.0e-14);
        assert(repair.moment_repaired_cells == 1);
        assert(repair.moment_repaired_rows == 1);
        assert(std::abs(static_cast<double>(std::accumulate(
            row.phase_energy_mass.begin(),
            row.phase_energy_mass.end(),
            0.0L) - energy_before)) < 1.0e-14);
        assert(std::abs(static_cast<double>(std::accumulate(
            row.phase_energy2_mass.begin(),
            row.phase_energy2_mass.end(),
            0.0L) - energy2_before)) < 1.0e-14);
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            const double mass = row.phase_mass[q];
            const double energy = row.phase_energy_mass[q];
            const double energy2 = row.phase_energy2_mass[q];
            assert(energy >= mass * row.energy_left - 1.0e-14);
            assert(energy <= mass * row.energy_right + 1.0e-14);
            assert(energy2 >= energy * energy / mass - 1.0e-14);
            assert(energy2 <= (row.energy_left + row.energy_right) * energy
                    - row.energy_left * row.energy_right * mass
                    + 1.0e-14);
        }

        ev::CanonicalOrbitRow over_budget;
        over_budget.energy_left = -1.0;
        over_budget.energy_right = 1.0;
        over_budget.phase_mass = {1.0e-14, 1.0, 1.0};
        over_budget.phase_energy_mass = {1.1e-14, -0.2, 0.2};
        over_budget.phase_energy2_mass = {1.21e-14, 0.3, 0.3};
        bool rejected = false;
        try {
            ev::CanonicalReversibleCollisionDiagnostics rejected_repair;
            ev::canonical_repair_row_energy_moments(
                over_budget, 0.5e-14, rejected_repair);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        assert(rejected);
    }

    ev::Params p;
    p.physics_model = "boltzmann_poisson";
    p.vp_universe = "static";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "power";
    p.kinetic_eta_power = 2.0;
    p.radial_grid = "log";
    p.inner_boundary = "delayed_return";
    p.r_min = 0.1;
    p.r_max = 7.0;
    p.w_max = 2.5;
    p.L_max = 2.0;
    p.Nr = 6;
    p.Nw = 8;
    p.NL = 6;
    p.bp_collision_operator = "boltzmann_reversible_sampled";
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 0.05;
    p.bp_boltzmann_pair_phi_points = 2;
    p.bp_boltzmann_scatter_mu_points = 2;
    p.bp_boltzmann_scatter_phi_points = 2;
    p.bp_reversible_candidate_radius = 4;
    p.bp_reversible_max_velocity_cells = p.Nw * p.NL;
    p.bp_boltzmann_remap_max_skipped_fraction = 0.9;
    p.bp_collision_fail_on_nonlocal_cells = false;
    p.bp_sampled_dense_cells = 8;
    p.bp_sampled_min_samples_per_batch = 256;
    p.bp_sampled_max_samples_per_batch = 256;
    p.bp_sampled_fail_on_unconverged = false;
    p.bp_reversible_active_shell_floor_fraction = 1.0e-8;
    p.threads = 4;

    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    ev::CanonicalAtlasBuildOptions options;
    options.phase_cells = 12;
    options.maximum_energy = 25.0;
    options.mapped_boundary_segments = 12;
    options.mapping_energy_points = 4;
    options.mapping_phase_points = 4;
    options.direct_tensor_quadrature = true;
    options.eta_quadrature_points = 2;
    options.stationary_distribution = [](double, double h) {
        return h < 4.0 ? 10.0 * std::exp(-h / 0.5) : 0.0;
    };
    options.energy_partition_weight = [](double, double h) {
        return h < 4.0 ? std::exp(-h) : 0.0;
    };
    auto construction = ev::build_canonical_bound_orbit_atlas(
        geometry,
        p,
        6,
        [](double r) { return 0.5 * r * r; },
        0.0,
        options);
    assert(ev::canonical_orbit_atlas_mass(construction.atlas) > 0.0);

    std::vector<ev::WeightedPICMarker> null_open;
    const auto null_step =
        ev::apply_canonical_reversible_collision_increment(
            construction.atlas,
            geometry,
            geometry.radial.center,
            geometry.radial_velocity.center,
            geometry.L,
            geometry.radial_velocity.width,
            geometry.legacy_Lq,
            1.0e-4,
            p,
            1.0e-6,
            1.0e-6,
            &null_open);
    assert(std::abs(null_step.physical_mass_residual) < 1.0e-11);
    assert(std::abs(null_step.physical_radial_momentum_residual) < 1.0e-10);
    assert(std::abs(null_step.physical_kinetic_energy_residual) < 1.0e-10);
    assert(std::abs(null_step.represented_mass_residual) < 1.0e-11);

    bool perturbed = false;
    for (ev::CanonicalOrbitRow& row : construction.atlas.row) {
        if (row.phase_mass.size() >= 2 && row.phase_mass[0] > 0.0) {
            const double transfer = 0.05 * row.phase_mass[0];
            row.phase_mass[0] -= transfer;
            row.phase_mass[1] += transfer;
            const double mean_h = row.phase_energy_mass[0]
                / (row.phase_mass[0] + transfer);
            row.phase_energy_mass[0] -= transfer * mean_h;
            row.phase_energy_mass[1] += transfer * mean_h;
            row.phase_energy2_mass[0] -= transfer * mean_h * mean_h;
            row.phase_energy2_mass[1] += transfer * mean_h * mean_h;
            perturbed = true;
            break;
        }
    }
    assert(perturbed);
    std::vector<ev::WeightedPICMarker> perturbed_open;
    const auto perturbed_step =
        ev::apply_canonical_reversible_collision_increment(
            construction.atlas,
            geometry,
            geometry.radial.center,
            geometry.radial_velocity.center,
            geometry.L,
            geometry.radial_velocity.width,
            geometry.legacy_Lq,
            1.0e-4,
            p,
            1.0e-6,
            1.0e-6,
            &perturbed_open);
    assert(std::abs(perturbed_step.physical_mass_residual) < 1.0e-11);
    assert(std::abs(perturbed_step.physical_radial_momentum_residual) < 1.0e-10);
    assert(std::abs(perturbed_step.physical_kinetic_energy_residual) < 1.0e-10);
    assert(std::abs(perturbed_step.represented_mass_residual) < 1.0e-11);
    assert(perturbed_step.minimum_atlas_cell_mass >= 0.0);
    long double open_mass = 0.0L;
    for (const ev::WeightedPICMarker& marker : perturbed_open) {
        open_mass += marker.mass;
    }
    assert(std::abs(static_cast<double>(open_mass)
            - 8.0 * ev::PI * ev::PI
                * perturbed_step.escaped_positive_increment_mass)
        < 1.0e-12);
    std::cout << "canonical reversible collision: null_l1="
              << null_step.eulerian_increment_l1
              << " perturbed_l1=" << perturbed_step.eulerian_increment_l1
              << " atlas_delta_l1=" << perturbed_step.atlas_increment_l1
              << " mass_error=" << perturbed_step.represented_mass_residual
              << " escaped=" << perturbed_step.escaped_increment_mass
              << '\n';
    return 0;
}
