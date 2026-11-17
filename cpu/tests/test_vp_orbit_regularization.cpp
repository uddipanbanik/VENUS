// Regression suite for ordinary versus Levi--Civita VP orbit integration:
// energy error, retry selection, pericentre resolution, and reflecting walls.

#include "ev/vlasov_poisson.hpp"
#include "ev/finite_angle_boltzmann.hpp"

#include <cassert>
#include <iostream>

namespace {

double kepler_energy(const ev::CharacteristicState& y, double angular_momentum) {
    return 0.5 * (
        y.w * y.w
        + (angular_momentum / y.r) * (angular_momentum / y.r))
        - 1.0 / y.r;
}

void test_generic_levi_civita_kepler_orbit() {
    constexpr double eccentricity = 0.999;
    const double angular_momentum =
        std::sqrt(1.0 - eccentricity * eccentricity);
    const ev::CharacteristicState initial{1.0 + eccentricity, 0.0};
    const double period = 2.0 * ev::PI;
    auto rhs = [&](const ev::CharacteristicState& y) {
        const double r = std::max(y.r, 1.0e-300);
        return ev::CharacteristicDerivative{
            y.w,
            angular_momentum * angular_momentum / (r * r * r)
                - 1.0 / (r * r),
        };
    };

    const ev::RegularizedOrbitResult result =
        ev::advance_radial_characteristic_levi_civita(
            initial, period, rhs, 1.0e-11, 1.0e-13, 4096);
    assert(result.converged);
    const double relative_energy_error = std::abs(
        (kepler_energy(result.state, angular_momentum)
            - kepler_energy(initial, angular_momentum))
        / kepler_energy(initial, angular_momentum));
    std::cout << "generic LC: attempted=" << result.attempted_steps
              << " accepted=" << result.accepted_steps
              << " dE/E=" << relative_energy_error
              << " dr=" << (result.state.r - initial.r)
              << " w=" << result.state.w << '\n';
    assert(relative_energy_error < 5.0e-8);
    assert(std::abs(result.state.r - initial.r) < 2.0e-7);
    assert(std::abs(result.state.w) < 2.0e-7);
}

void test_vp_retry_beats_one_step_kdk() {
    ev::vp_orbit_lc_retries.store(0, std::memory_order_relaxed);
    ev::vp_orbit_lc_accepted.store(0, std::memory_order_relaxed);
    ev::vp_orbit_lc_unresolved.store(0, std::memory_order_relaxed);
    ev::vp_orbit_maximum_ordinary_energy_error.store(
        0.0, std::memory_order_relaxed);
    ev::vp_orbit_maximum_final_energy_error.store(
        0.0, std::memory_order_relaxed);

    ev::Params p;
    p.physics_model = "vlasov_poisson";
    p.vp_universe = "static";
    p.vp_gravity = 1.0;
    p.central_mass = 1.0;
    p.inner_boundary = "origin_parity";
    p.outer_boundary = "outflow_vacuum";
    p.radial_grid = "log";
    p.r_min = 1.0e-5;
    p.r_max = 4.0;
    p.integrator = "semi_lagrangian_kick_drift_kick";
    p.vp_orbit_regularization = "auto";
    p.vp_orbit_energy_tolerance = 1.0e-5;
    p.vp_orbit_lc_relative_tolerance = 1.0e-10;
    p.vp_orbit_lc_absolute_tolerance = 1.0e-13;
    p.vp_orbit_lc_max_steps = 4096;
    p.vp_orbit_lc_max_refinements = 1;

    constexpr int points = 512;
    std::vector<double> r(points);
    std::vector<double> x(points);
    ev::VlasovPoissonField field;
    field.phi.resize(points);
    field.phi_r.resize(points);
    field.source_mass.assign(points, 1.0);
    for (int i = 0; i < points; ++i) {
        const double fraction = static_cast<double>(i) / static_cast<double>(points - 1);
        r[static_cast<std::size_t>(i)] =
            p.r_min * std::pow(p.r_max / p.r_min, fraction);
        x[static_cast<std::size_t>(i)] =
            std::log(r[static_cast<std::size_t>(i)]);
        field.phi[static_cast<std::size_t>(i)] =
            -1.0 / r[static_cast<std::size_t>(i)];
        field.phi_r[static_cast<std::size_t>(i)] =
            1.0 / (r[static_cast<std::size_t>(i)] * r[static_cast<std::size_t>(i)]);
    }

    constexpr double eccentricity = 0.995;
    const double angular_momentum =
        std::sqrt(1.0 - eccentricity * eccentricity);
    const ev::CharacteristicState initial{1.0 + eccentricity, 0.0};
    const double period = 2.0 * ev::PI;
    const ev::CharacteristicIntegratorPlan plan =
        ev::make_characteristic_integrator_plan(p);
    const ev::CharacteristicState ordinary =
        ev::advance_vp_characteristic_unregularized(
            initial.r,
            initial.w,
            angular_momentum,
            x,
            r,
            field,
            period,
            p,
            plan);
    const ev::CharacteristicState regularized =
        ev::advance_vp_characteristic(
            initial.r,
            initial.w,
            angular_momentum,
            x,
            r,
            field,
            period,
            p,
            plan);
    auto represented_rhs = [&](const ev::CharacteristicState& y) {
        const ev::VPSample sample =
            ev::sample_vp_field(y.r, x, r, field, p);
        return ev::CharacteristicDerivative{
            y.w,
            angular_momentum * angular_momentum / (y.r * y.r * y.r)
                - sample.phi_r,
        };
    };
    const ev::RegularizedOrbitResult direct =
        ev::advance_radial_characteristic_levi_civita(
            initial,
            period,
            represented_rhs,
            p.vp_orbit_lc_relative_tolerance,
            p.vp_orbit_lc_absolute_tolerance,
            p.vp_orbit_lc_max_steps);
    const double direct_hamiltonian_error =
        ev::vp_frozen_hamiltonian_error(
            initial, direct.state, angular_momentum, x, r, field, p);
    const double ordinary_error =
        std::abs(kepler_energy(ordinary, angular_momentum)
            - kepler_energy(initial, angular_momentum));
    const double regularized_error =
        std::abs(kepler_energy(regularized, angular_momentum)
            - kepler_energy(initial, angular_momentum));
    const double represented_hamiltonian_error =
        ev::vp_frozen_hamiltonian_error(
            initial, regularized, angular_momentum, x, r, field, p);
    std::cout << "VP retry: ordinary dE=" << ordinary_error
              << " regularized dE=" << regularized_error
              << " represented dE/E=" << represented_hamiltonian_error
              << " direct_converged=" << direct.converged
              << " direct_steps=" << direct.attempted_steps
              << " direct_dE/E=" << direct_hamiltonian_error
              << " r=" << regularized.r
              << " w=" << regularized.w << '\n';
    assert(represented_hamiltonian_error < p.vp_orbit_energy_tolerance);
    assert(regularized_error < 1.0e-5);
    assert(regularized_error < 1.0e-4 * ordinary_error);

    const ev::CharacteristicState pericentre =
        ev::advance_vp_characteristic(
            initial.r,
            initial.w,
            angular_momentum,
            x,
            r,
            field,
            0.5 * period,
            p,
            plan);
    const ev::CharacteristicState round_trip =
        ev::advance_vp_characteristic(
            pericentre.r,
            pericentre.w,
            angular_momentum,
            x,
            r,
            field,
            -0.5 * period,
            p,
            plan);
    const double round_trip_error = std::hypot(
        (round_trip.r - initial.r) / initial.r,
        round_trip.w);
    const ev::VPOrbitRegularizationDiagnostics diagnostics =
        ev::vp_orbit_regularization_diagnostics();
    std::cout << "VP LC round trip: error=" << round_trip_error
              << " max_ordinary_dE/E="
              << diagnostics.maximum_ordinary_energy_error
              << " max_final_dE/E="
              << diagnostics.maximum_final_energy_error << '\n';
    assert(round_trip_error < 3.0e-5);
    assert(
        diagnostics.maximum_ordinary_energy_error
        > p.vp_orbit_energy_tolerance);
    assert(
        diagnostics.maximum_final_energy_error
        <= p.vp_orbit_energy_tolerance);
    assert(diagnostics.unresolved == 0);
}

void test_origin_potential_and_force_are_consistent() {
    ev::Params p;
    p.physics_model = "vlasov_poisson";
    p.inner_boundary = "origin_parity";
    p.central_mass = 0.0;
    std::vector<double> r{0.1, 0.2, 0.4};
    std::vector<double> x = r;
    ev::VlasovPoissonField field;
    field.phi = {0.005, 0.020, 0.080};
    field.phi_r = {0.1, 0.2, 0.4};
    field.source_mass = {0.001, 0.008, 0.064};
    for (double rp : {0.0, 0.025, 0.075, 0.15, 0.3}) {
        const ev::VPSample sample =
            ev::sample_vp_field(rp, x, r, field, p);
        assert(std::abs(sample.phi - 0.5 * rp * rp) < 2.0e-14);
        assert(std::abs(sample.phi_r - rp) < 2.0e-14);
    }
}

void test_isotropic_tangential_moment() {
    ev::Params p;
    p.Nr = 1;
    p.Nw = 201;
    p.NL = 201;
    p.threads = 1;
    const std::vector<double> r{1.0};
    const std::vector<double> w = ev::linspace(-5.0, 5.0, p.Nw);
    const std::vector<double> L = ev::linspace(0.0, 5.0, p.NL);
    const std::vector<double> wq = ev::quadrature_weights(w);
    const std::vector<double> Lq = ev::quadrature_weights(L);
    std::vector<double> f(static_cast<std::size_t>(p.Nr) * p.Nw * p.NL);
    for (int j = 0; j < p.Nw; ++j) {
        for (int k = 0; k < p.NL; ++k) {
            f[ev::idx(0, j, k, p)] = std::exp(-w[j] * w[j] - L[k] * L[k]);
        }
    }
    const ev::Moments moments = ev::compute_vp_moments(f, r, w, L, wq, Lq, p);
    const double ratio = moments.pt[0] / moments.p[0];
    std::cout << "isotropic moment: pt/p=" << ratio << '\n';
    assert(std::abs(ratio - 2.0) < 2.0e-3);
}

void test_boltzmann_cumulative_mass_tolerance() {
    ev::Params p;
    p.Nr = 1;
    p.Nw = 2;
    p.NL = 4;
    p.bp_boltzmann_mass_tol = 0.05;
    const std::vector<double> r{1.0};
    const std::vector<double> w{-1.0, 1.0};
    const std::vector<double> L{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> wq = ev::quadrature_weights(w);
    const std::vector<double> Lq = ev::quadrature_weights(L);
    const std::vector<double> volume =
        ev::bp_fa_shell_velocity_volumes(r[0], wq, L, Lq, p);
    const double desired_n[] = {0.02, 0.02, 0.02, 0.02, 0.02, 0.90};
    std::vector<double> f(static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 0.0);
    int q = 0;
    for (int j = 0; j < p.Nw; ++j) {
        for (int k = 1; k < p.NL; ++k) {
            const int shell_index = j * p.NL + k;
            f[ev::idx(0, j, k, p)] =
                desired_n[q++] / volume[static_cast<std::size_t>(shell_index)];
        }
    }
    const std::vector<ev::BPFAVelocityCell> cells =
        ev::bp_fa_active_cells(f, 0, r[0], w, L, volume, p);
    double kept_n = 0.0;
    for (const auto& cell : cells) {
        kept_n += cell.n;
    }
    std::cout << "Boltzmann tail budget: kept=" << kept_n
              << " omitted=" << (1.0 - kept_n) << '\n';
    assert(cells.size() == 4);
    assert(std::abs(kept_n - 0.96) < 1.0e-14);
    assert(1.0 - kept_n <= p.bp_boltzmann_mass_tol + 1.0e-14);
}

void test_frank_core_collision_rate_units() {
    constexpr double sigma_physical = 25.0;
    constexpr double sigma_scale_physical = 9.71;
    constexpr double rho_over_rho_s = 2.5;
    constexpr double u_over_v_s = 1.7;
    const double sigma_tilde = sigma_physical / sigma_scale_physical;
    const double rho_code = rho_over_rho_s / (4.0 * ev::PI);
    const double sigma_code = 4.0 * ev::PI * sigma_tilde;
    const double rate_code = rho_code * sigma_code * u_over_v_s;
    const double rate_frank = rho_over_rho_s * sigma_tilde * u_over_v_s;
    assert(std::abs(rate_code - rate_frank) < 1.0e-14 * rate_frank);

    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = sigma_code;
    const std::vector<ev::BPFAVelocityCell> cells{
        ev::BPFAVelocityCell{0, 0, 0, 0.0, 0.0, 1.0, 2.0},
        ev::BPFAVelocityCell{1, 0, 1, 1.0, 0.0, 1.0, 3.0},
    };
    std::vector<double> phi_cos;
    std::vector<double> phi_sin;
    ev::bp_fa_midpoint_angle_table(4, phi_cos, phi_sin);
    const double max_loss_rate =
        ev::bp_fa_shell_max_loss_rate(cells, phi_cos, phi_sin, p, 1);
    const double expected_max_loss_rate = 3.0 * sigma_code;
    std::cout << "Frank Core units: sigma_code=" << sigma_code
              << " rate_identity=" << rate_code
              << " max_loss_rate=" << max_loss_rate << '\n';
    assert(std::abs(max_loss_rate - expected_max_loss_rate)
           < 1.0e-14 * expected_max_loss_rate);
}

void test_constant_cross_section_loss_bound() {
    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_collision_strength = 0.7;
    p.bp_sigma0 = 3.2;
    const std::vector<ev::BPFAVelocityCell> cells{
        ev::BPFAVelocityCell{0, 0, 0, -1.1, 0.2, 1.0, 0.8},
        ev::BPFAVelocityCell{1, 0, 1, 0.3, 0.9, 1.0, 1.7},
        ev::BPFAVelocityCell{2, 0, 2, 1.4, 0.5, 1.0, 0.4},
        ev::BPFAVelocityCell{3, 0, 3, -0.2, 1.6, 1.0, 2.1},
    };
    std::vector<double> phi_cos;
    std::vector<double> phi_sin;
    ev::bp_fa_midpoint_angle_table(32, phi_cos, phi_sin);
    const double exact =
        ev::bp_fa_shell_max_loss_rate(cells, phi_cos, phi_sin, p, 1);
    const double bound =
        ev::bp_fa_constant_shell_loss_rate_upper_bound(cells, p);
    std::cout << "constant-sigma loss bound: exact=" << exact
              << " bound=" << bound << '\n';
    assert(bound >= exact);
}

void test_gauss_legendre_mu_quadrature() {
    for (int points = 1; points <= 8; ++points) {
        std::vector<double> nodes;
        std::vector<double> weights;
        ev::bp_fa_gauss_legendre_mu_table(points, nodes, weights);
        double average_one = 0.0;
        double average_mu = 0.0;
        double average_mu2 = 0.0;
        for (int q = 0; q < points; ++q) {
            average_one += weights[static_cast<std::size_t>(q)];
            average_mu += weights[static_cast<std::size_t>(q)]
                * nodes[static_cast<std::size_t>(q)];
            average_mu2 += weights[static_cast<std::size_t>(q)]
                * nodes[static_cast<std::size_t>(q)]
                * nodes[static_cast<std::size_t>(q)];
        }
        assert(std::abs(average_one - 1.0) < 2.0e-14);
        assert(std::abs(average_mu) < 2.0e-14);
        if (points >= 2) {
            assert(std::abs(average_mu2 - 1.0 / 3.0) < 2.0e-14);
        }
    }
}

void test_boltzmann_remap_overflow_diagnostics() {
    ev::Params p;
    p.Nw = 3;
    p.NL = 3;
    const std::vector<double> w{-1.0, 0.0, 1.0};
    const std::vector<double> L{0.0, 1.0, 2.0};
    const std::vector<double> eta{0.0, 0.5, 2.0};
    const std::vector<double> volume(9, 1.0);
    const ev::BPFABracketLookup w_lookup = ev::bp_fa_make_bracket_lookup(w, 0);
    const ev::BPFABracketLookup eta_lookup = ev::bp_fa_make_bracket_lookup(eta, 1);
    ev::BPFARemapStencil stencil;
    bool w_outside = false;
    bool L_outside = false;
    assert(ev::bp_fa_remap_stencil(
        {0.25, 1.0, 0.0},
        1.0,
        w,
        eta,
        w_lookup,
        eta_lookup,
        volume,
        p,
        stencil,
        &w_outside,
        &L_outside));
    assert(!w_outside && !L_outside);
    assert(!ev::bp_fa_remap_stencil(
        {1.5, 1.0, 0.0},
        1.0,
        w,
        eta,
        w_lookup,
        eta_lookup,
        volume,
        p,
        stencil,
        &w_outside,
        &L_outside));
    assert(w_outside && !L_outside);
    assert(!ev::bp_fa_remap_stencil(
        {0.25, 3.0, 0.0},
        1.0,
        w,
        eta,
        w_lookup,
        eta_lookup,
        volume,
        p,
        stencil,
        &w_outside,
        &L_outside));
    assert(!w_outside && L_outside);
}

void test_frank_dynamical_timestep_is_not_a_cell_crossing_cfl() {
    ev::Params p;
    p.physics_model = "vlasov_poisson";
    p.Nr = 1;
    p.Nw = 1;
    p.NL = 1;
    p.threads = 1;
    p.dt_max = 10.0;
    p.cfl = 0.35;
    p.vp_dynamical_eta = 0.01;
    p.active_floor_abs = 0.0;
    p.active_floor_rel = 0.0;
    const std::vector<double> f{1.0};
    const std::vector<double> r{2.0};
    const std::vector<double> w{0.3};
    const std::vector<double> L{1.0};
    ev::VlasovPoissonField field;
    field.phi = {-0.4};
    field.phi_r = {0.2};
    field.source_mass = {1.0};
    field.background = ev::vp_background(0.0, p);
    const ev::MpiContext mpi;

    p.vp_timestep_mode = "active_cell_cfl";
    const double cell_cfl_dt = ev::choose_dt_vlasov_poisson(
        f, r, w, L, field, 0.01, 0.01, p, mpi);
    p.vp_timestep_mode = "frank_dynamical";
    const double dynamical_dt = ev::choose_dt_vlasov_poisson(
        f, r, w, L, field, 0.01, 0.01, p, mpi);
    const double acceleration = std::abs(1.0 / 8.0 - 0.2);
    const double expected =
        std::sqrt(2.0 * p.vp_dynamical_eta * r[0] / acceleration);
    assert(std::abs(dynamical_dt - expected) < 1.0e-14 * expected);
    assert(dynamical_dt > 10.0 * cell_cfl_dt);
}

double maxwellian_collision_remap_rate(
    int radial_velocity_points,
    int angular_momentum_points,
    int pair_phi_points,
    int scatter_mu_points,
    int scatter_phi_points) {
    ev::Params p;
    p.Nr = 1;
    p.Nw = radial_velocity_points;
    p.NL = angular_momentum_points;
    p.threads = 1;
    p.bp_cross_section = "constant";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 1.0;
    p.bp_boltzmann_pair_phi_points = pair_phi_points;
    p.bp_boltzmann_scatter_mu_points = scatter_mu_points;
    p.bp_boltzmann_scatter_phi_points = scatter_phi_points;
    p.bp_boltzmann_mass_tol = 0.0;
    p.bp_boltzmann_conserve_moments = true;
    const std::vector<double> r{1.0};
    const std::vector<double> w =
        ev::stretched_w_grid(4.0, 3.5, 0.0, p.Nw);
    std::vector<double> L(static_cast<std::size_t>(p.NL), 0.0);
    for (int k = 0; k < p.NL; ++k) {
        const double fraction =
            static_cast<double>(k) / static_cast<double>(p.NL - 1);
        L[static_cast<std::size_t>(k)] = 4.0 * std::pow(fraction, 5.0);
    }
    const std::vector<double> wq = ev::quadrature_weights(w);
    const std::vector<double> Lq = ev::quadrature_weights(L);
    const std::vector<double> volume =
        ev::bp_fa_shell_velocity_volumes(r[0], wq, L, Lq, p);
    std::vector<double> f(static_cast<std::size_t>(p.Nw * p.NL), 0.0);
    for (int j = 0; j < p.Nw; ++j) {
        for (int k = 0; k < p.NL; ++k) {
            f[ev::idx(0, j, k, p)] =
                std::exp(-0.5 * (w[j] * w[j] + L[k] * L[k]));
        }
    }
    const ev::BPFAShellMoments unnormalized =
        ev::bp_fa_shell_moments(f, 0, r[0], w, L, volume, p);
    for (double& value : f) {
        value /= unnormalized.mass;
    }
    const std::vector<double> initial = f;
    const ev::BPFAShellMoments before =
        ev::bp_fa_shell_moments(f, 0, r[0], w, L, volume, p);
    constexpr double dt = 1.0e-3;
    const ev::BPFARemapBudget remap =
        ev::bp_fa_apply_shell_substep(f, 0, dt, r, w, L, wq, Lq, p, 1);
    const ev::BPFAShellMoments after =
        ev::bp_fa_shell_moments(f, 0, r[0], w, L, volume, p);
    double weighted_l1 = 0.0;
    for (int j = 0; j < p.Nw; ++j) {
        for (int k = 0; k < p.NL; ++k) {
            const int shell_index = j * p.NL + k;
            weighted_l1 += volume[static_cast<std::size_t>(shell_index)]
                * std::abs(f[ev::idx(0, j, k, p)] - initial[ev::idx(0, j, k, p)]);
        }
    }
    assert(std::abs(after.mass - before.mass) < 1.0e-12 * before.mass);
    assert(std::abs(after.radial_momentum - before.radial_momentum)
           < 1.0e-12 * before.mass);
    assert(std::abs(after.kinetic_energy - before.kinetic_energy)
           < 1.0e-12 * before.kinetic_energy);
    assert(remap.skipped_event_density <= remap.attempted_event_density);
    return weighted_l1 / (before.mass * dt);
}

void test_maxwellian_collision_residual_converges_with_angular_quadrature() {
    const double aliased_222_rate =
        maxwellian_collision_remap_rate(14, 20, 2, 2, 2);
    const double gauss_424_rate =
        maxwellian_collision_remap_rate(14, 20, 4, 2, 4);
    const double tensor_444_rate =
        maxwellian_collision_remap_rate(14, 20, 4, 4, 4);
    const double tensor_888_rate =
        maxwellian_collision_remap_rate(14, 20, 8, 8, 8);
    std::cout << "Maxwellian collision remap: q222=" << aliased_222_rate
              << " q424=" << gauss_424_rate
              << " q444=" << tensor_444_rate
              << " q888=" << tensor_888_rate << '\n';
    assert(gauss_424_rate < 0.4 * aliased_222_rate);
    assert(tensor_444_rate <= 1.1 * gauss_424_rate);
    assert(tensor_888_rate < tensor_444_rate);
    assert(tensor_888_rate < 0.2);
}

void test_p2_line_mass_poisson_reconstruction() {
    ev::Params p;
    p.physics_model = "vlasov_poisson";
    p.vp_universe = "static";
    p.vp_poisson_source = "total_density";
    p.vp_poisson_reconstruction = "p2_line_mass";
    p.vp_interpolation = "hamiltonian_sldg";
    p.kinetic_grid = "cell_average_eta";
    p.radial_grid = "log";
    p.Nr = 12;
    p.r_min = 0.2;
    p.r_max = 4.0;
    p.central_mass = 0.7;
    const double hidden_mass = 0.2;
    const ev::PhaseSpaceAxis radial = ev::make_positive_radial_cell_axis(p);
    ev::Moments moments;
    moments.rho.assign(static_cast<std::size_t>(p.Nr), 0.0);
    moments.p.assign(static_cast<std::size_t>(p.Nr), 0.0);
    moments.pt.assign(static_cast<std::size_t>(p.Nr), 0.0);
    moments.j.assign(static_cast<std::size_t>(p.Nr), 0.0);
    auto primitive = [](long double radius) {
        return radius + 0.1L * radius * radius
            + 0.01L * radius * radius * radius;
    };
    for (int i = 0; i < p.Nr; ++i) {
        const long double left = radial.edge[static_cast<std::size_t>(i)];
        const long double right = radial.edge[static_cast<std::size_t>(i + 1)];
        const long double q_average =
            (primitive(right) - primitive(left)) / (right - left);
        const long double center = radial.center[static_cast<std::size_t>(i)];
        moments.rho[static_cast<std::size_t>(i)] = static_cast<double>(
            q_average / (4.0L * ev::PI * center * center));
    }
    const ev::VlasovPoissonField field = ev::solve_vlasov_poisson_field(
        moments, radial.center, 0.0, p, hidden_mass);
    const long double source_origin = primitive(p.r_min);
    auto exact_mass = [&](long double radius) {
        return static_cast<long double>(p.central_mass + hidden_mass)
            + primitive(radius) - source_origin;
    };
    const long double outer_mass = exact_mass(p.r_max);
    auto exact_force_integral = [&](long double lower, long double upper) {
        const long double constant =
            static_cast<long double>(p.central_mass + hidden_mass)
            - source_origin;
        return constant * (1.0L / lower - 1.0L / upper)
            + std::log(upper / lower)
            + 0.1L * (upper - lower)
            + 0.005L * (upper * upper - lower * lower);
    };
    double maximum_mass_error = 0.0;
    double maximum_force_error = 0.0;
    double maximum_potential_error = 0.0;
    for (int i = 0; i < p.Nr; ++i) {
        for (double fraction : {0.17, 0.53, 0.89}) {
            const double radius = std::fma(
                fraction,
                radial.edge[static_cast<std::size_t>(i + 1)]
                    - radial.edge[static_cast<std::size_t>(i)],
                radial.edge[static_cast<std::size_t>(i)]);
            const ev::VPSample sample = ev::sample_vp_field(
                radius,
                radial.center,
                radial.center,
                field,
                p,
                i);
            const long double mass = exact_mass(radius);
            const long double force = mass / (radius * radius);
            const long double potential = -outer_mass / p.r_max
                - exact_force_integral(radius, p.r_max);
            maximum_force_error = std::max(
                maximum_force_error,
                std::abs(sample.phi_r - static_cast<double>(force)));
            maximum_potential_error = std::max(
                maximum_potential_error,
                std::abs(sample.phi - static_cast<double>(potential)));
            const double represented_mass = sample.phi_r * radius * radius;
            maximum_mass_error = std::max(
                maximum_mass_error,
                std::abs(represented_mass - static_cast<double>(mass)));
        }
    }
    const double mass_scale = std::max(1.0, static_cast<double>(outer_mass));
    std::cout << "P2 Poisson: outer_mass_error="
              << field.outer_source_mass - static_cast<double>(outer_mass)
              << " mass_error=" << maximum_mass_error
              << " force_error=" << maximum_force_error
              << " potential_error=" << maximum_potential_error << std::endl;
    assert(std::abs(field.outer_source_mass - static_cast<double>(outer_mass))
        < 2.0e-13 * mass_scale);
    assert(maximum_mass_error < 2.0e-12 * mass_scale);
    assert(maximum_force_error < 2.0e-11 * mass_scale);
    assert(maximum_potential_error < 2.0e-11 * mass_scale);
}

}  // namespace

int main() {
    test_generic_levi_civita_kepler_orbit();
    test_vp_retry_beats_one_step_kdk();
    test_origin_potential_and_force_are_consistent();
    test_isotropic_tangential_moment();
    test_boltzmann_cumulative_mass_tolerance();
    test_frank_core_collision_rate_units();
    test_constant_cross_section_loss_bound();
    test_gauss_legendre_mu_quadrature();
    test_boltzmann_remap_overflow_diagnostics();
    test_frank_dynamical_timestep_is_not_a_cell_crossing_cfl();
    test_maxwellian_collision_residual_converges_with_angular_quadrature();
    test_p2_line_mass_poisson_reconstruction();
    std::cout << "VP Levi-Civita orbit tests passed\n";
    return 0;
}
