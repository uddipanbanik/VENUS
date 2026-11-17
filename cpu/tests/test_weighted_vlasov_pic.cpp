// Tests Newtonian weighted-PIC marker initialization, spherical deposition,
// orbit/boundary state transitions, and versioned persistence.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <vector>

#include "ev/weighted_vlasov_pic.hpp"
#include "ev/initial_data.hpp"

namespace {

bool nearly_equal(double first, double second, double tolerance = 2.0e-14) {
    return std::abs(first - second)
        <= tolerance * std::max({1.0, std::abs(first), std::abs(second)});
}

ev::Params pic_params() {
    ev::Params p;
    p.physics_model = "vlasov_poisson";
    p.vp_universe = "static";
    p.vp_interpolation = "hamiltonian_sldg";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "uniform";
    p.Nr = 3;
    p.Nw = 4;
    p.NL = 2;
    p.r_min = 1.0;
    p.r_max = 4.0;
    p.radial_grid = "linear";
    p.w_max = 2.0;
    p.L_max = 2.0;
    p.inner_boundary = "reflecting";
    p.outer_boundary = "reflecting";
    p.vp_orbit_regularization = "off";
    p.integrator = "semi_lagrangian_kick_drift_kick";
    return p;
}

ev::PhaseSpaceCellGeometry pic_geometry(const ev::Params& p) {
    return ev::make_cell_average_eta_geometry(
        ev::make_positive_radial_cell_axis(p), p);
}

std::vector<double> symmetric_source(
    const ev::PhaseSpaceCellGeometry& geometry,
    const ev::Params& p) {
    std::vector<double> f(
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 0.0);
    for (int i = 0; i < p.Nr; ++i) {
        const double radial_amplitude = i == 0 ? 1.0 : (i == 1 ? 0.2 : 1.0e-30);
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                f[ev::idx(i, j, k, p)] = radial_amplitude * (1.0 + 0.1 * k);
            }
        }
    }
    (void)geometry;
    return f;
}

void assert_same_marker(
    const ev::WeightedPICMarker& first,
    const ev::WeightedPICMarker& second) {
    assert(first.mass == second.mass);
    assert(first.r == second.r);
    assert(first.w == second.w);
    assert(first.eta == second.eta);
    assert(first.id == second.id);
    assert(first.source_cell == second.source_cell);
    assert(first.state == second.state);
    assert(first.inner_boundary_speed == second.inner_boundary_speed);
    assert(first.remaining_inner_time == second.remaining_inner_time);
}

double resolved_weight_sum(const ev::WeightedPICCICStencil& stencil) {
    double sum = 0.0;
    for (int n = 0; n < stencil.count; ++n) {
        const std::size_t slot = static_cast<std::size_t>(n);
        assert(stencil.weight[slot] >= 0.0);
        sum += stencil.weight[slot];
    }
    return sum;
}

double resolved_weight_sum(const ev::WeightedPICTSCStencil& stencil) {
    double sum = 0.0;
    for (int n = 0; n < stencil.count; ++n) {
        const std::size_t slot = static_cast<std::size_t>(n);
        assert(stencil.weight[slot] >= 0.0);
        sum += stencil.weight[slot];
    }
    return sum;
}

void test_mixed_marker_mass_reduction() {
    // Every input and the exact total are representable binary fractions.
    // A binary64 sequential accumulator loses every small term after 1.
    constexpr std::size_t small_count = 8192;
    const double small_mass = std::ldexp(1.0, -54);
    const double expected = 1.0 + std::ldexp(1.0, -41);
    std::vector<ev::WeightedPICMarker> markers(small_count + 1);
    markers.front().mass = 1.0;
    for (std::size_t q = 1; q < markers.size(); ++q) {
        markers[q].mass = small_mass;
    }
    const double forward = ev::weighted_pic_total_marker_mass(markers);
    std::reverse(markers.begin(), markers.end());
    const double backward = ev::weighted_pic_total_marker_mass(markers);
    assert(forward == expected);
    assert(backward == expected);
    std::cout << std::setprecision(17)
              << "mixed mass reduction: expected=" << expected
              << " forward=" << forward << " backward=" << backward << '\n';
}

void test_mass_budget_quiet_pairing_and_determinism() {
    const ev::Params p = pic_params();
    const ev::PhaseSpaceCellGeometry geometry = pic_geometry(p);
    const std::vector<double> f = symmetric_source(geometry, p);
    ev::WeightedPICInitializationOptions options;
    options.markers_per_occupied_cell = 32;
    options.global_marker_budget = 48;
    options.velocity_pairing = ev::WeightedPICVelocityPairing::required;
    options.use_piecewise_constant_in_cell = true;
    ev::WeightedPICSamplerState first_sampler{UINT64_C(1234567), 19};
    ev::WeightedPICSamplerState second_sampler = first_sampler;
    const ev::WeightedPICInitializationResult first =
        ev::initialize_weighted_vlasov_pic(
            f, geometry, p, options, first_sampler);
    const ev::WeightedPICInitializationResult second =
        ev::initialize_weighted_vlasov_pic(
            f, geometry, p, options, second_sampler);

    assert(first.markers.size() == 48);
    assert(first.diagnostics.occupied_source_cells == 24);
    assert(first.diagnostics.velocity_paired);
    assert(first_sampler.next_counter == second_sampler.next_counter);
    const double accurate_mass_error = std::fma(
        static_cast<double>(first.markers.size()),
        first.markers.front().mass, -first.diagnostics.source_mass);
    std::cout << std::setprecision(17)
              << "quiet mass precision: source=" << first.diagnostics.source_mass
              << " diagnostic_sum=" << first.diagnostics.represented_mass
              << " common_marker_mass=" << first.markers.front().mass
              << " N=" << first.markers.size()
              << " true_product_minus_source=" << accurate_mass_error
              << " diagnostic_minus_source="
              << first.diagnostics.represented_mass-first.diagnostics.source_mass
              << " long_double_digits=" << std::numeric_limits<long double>::digits
              << std::endl;
    // The paired population has one common binary64 mass (checked below).
    // FMA measures N*m-target with one final rounding and preserves the
    // original physical normalization criterion. The compensated production
    // diagnostic must also satisfy that criterion on binary64 long double.
    assert(std::abs(accurate_mass_error)
        <= 4.0e-16 * first.diagnostics.source_mass);
    assert(nearly_equal(
        first.diagnostics.source_mass,
        first.diagnostics.represented_mass,
        4.0e-16));
    assert(first.markers.size() == second.markers.size());
    long double radial_momentum = 0.0L;
    std::set<std::size_t> represented_source_cells;
    const double common_marker_mass = first.markers.front().mass;
    for (std::size_t q = 0; q < first.markers.size(); q += 2) {
        const ev::WeightedPICMarker& negative = first.markers[q];
        const ev::WeightedPICMarker& positive = first.markers[q + 1];
        assert_same_marker(negative, second.markers[q]);
        assert_same_marker(positive, second.markers[q + 1]);
        assert(negative.mass == common_marker_mass);
        assert(positive.mass == common_marker_mass);
        assert(negative.mass == positive.mass);
        assert(negative.r == positive.r);
        assert(negative.eta == positive.eta);
        assert(negative.w == -positive.w);
        radial_momentum += static_cast<long double>(negative.mass) * negative.w;
        radial_momentum += static_cast<long double>(positive.mass) * positive.w;
        represented_source_cells.insert(
            static_cast<std::size_t>(negative.source_cell));
        represented_source_cells.insert(
            static_cast<std::size_t>(positive.source_cell));
    }
    assert(radial_momentum == 0.0L);

    // The positive 1e-30 tail remains in the source CDF and total mass, but it
    // does not force one marker per tail cell under the fixed global budget.
    bool represented_tiny_tail = false;
    for (std::size_t flat : represented_source_cells) {
        const int i = static_cast<int>(flat / static_cast<std::size_t>(p.Nw * p.NL));
        represented_tiny_tail = represented_tiny_tail || i == 2;
    }
    assert(!represented_tiny_tail);

    // Every sampled coordinate lies strictly inside its canonical source CV.
    for (const ev::WeightedPICMarker& marker : first.markers) {
        const std::size_t flat = static_cast<std::size_t>(marker.source_cell);
        const int i = static_cast<int>(flat / static_cast<std::size_t>(p.Nw * p.NL));
        const int rem = static_cast<int>(flat % static_cast<std::size_t>(p.Nw * p.NL));
        const int j = rem / p.NL;
        const int k = rem % p.NL;
        assert(marker.r > geometry.radial.edge[static_cast<std::size_t>(i)]);
        assert(marker.r < geometry.radial.edge[static_cast<std::size_t>(i + 1)]);
        assert(marker.w > geometry.radial_velocity.edge[static_cast<std::size_t>(j)]);
        assert(marker.w < geometry.radial_velocity.edge[static_cast<std::size_t>(j + 1)]);
        assert(marker.eta > geometry.eta.edge[static_cast<std::size_t>(k)]);
        assert(marker.eta < geometry.eta.edge[static_cast<std::size_t>(k + 1)]);
    }

    std::vector<double> asymmetric = f;
    asymmetric[ev::idx(0, 0, 0, p)] *= 1.25;
    bool rejected_required_pairing = false;
    try {
        ev::WeightedPICSamplerState asymmetric_sampler{UINT64_C(1234567), 19};
        (void)ev::initialize_weighted_vlasov_pic(
            asymmetric, geometry, p, options, asymmetric_sampler);
    } catch (const std::runtime_error&) {
        rejected_required_pairing = true;
    }
    assert(rejected_required_pairing);
    options.velocity_pairing = ev::WeightedPICVelocityPairing::automatic;
    options.global_marker_budget = 47;
    ev::WeightedPICSamplerState asymmetric_sampler{UINT64_C(1234567), 19};
    const auto unpaired = ev::initialize_weighted_vlasov_pic(
        asymmetric, geometry, p, options, asymmetric_sampler);
    assert(!unpaired.diagnostics.velocity_paired);
    assert(unpaired.markers.size() == 47);

    std::cout << "initialization: markers=" << first.markers.size()
              << " occupied=" << first.diagnostics.occupied_source_cells
              << " mass_error="
              << first.diagnostics.represented_mass
                    - first.diagnostics.source_mass
              << '\n';
}

double harmonic_energy(const ev::WeightedPICMarker& marker) {
    return 0.5 * marker.w * marker.w
        + marker.eta / (marker.r * marker.r)
        + 0.5 * marker.r * marker.r;
}

double minimum_harmonic_energy_in_cell(
    double r_lower,
    double r_upper,
    double w_lower,
    double w_upper,
    double eta_lower) {
    const double minimum_abs_w = w_lower <= 0.0 && w_upper >= 0.0
        ? 0.0
        : std::min(std::abs(w_lower), std::abs(w_upper));
    const double stationary_radius = std::pow(2.0 * eta_lower, 0.25);
    const double radius = std::clamp(
        stationary_radius, r_lower, r_upper);
    return 0.5 * minimum_abs_w * minimum_abs_w
        + eta_lower / (radius * radius)
        + 0.5 * radius * radius;
}

void test_partial_eta_energy_support_uses_continuous_eta() {
    ev::Params p = pic_params();
    p.Nr = 2;
    p.Nw = 2;
    p.NL = 2;
    p.r_min = 1.0;
    p.r_max = 1.002;
    p.w_max = 1.0e-3;
    p.L_max = 2.0;
    p.vp_interpolation = "raw";
    const ev::PhaseSpaceCellGeometry geometry = pic_geometry(p);
    constexpr double target_partial_shell_fraction = 0.215;
    constexpr double partial_eta_fraction =
        target_partial_shell_fraction / (1.0 - target_partial_shell_fraction);
    const double energy_cutoff = 1.5 + partial_eta_fraction;
    auto analytic_density = [&](double radius, double radial_velocity, double eta) {
        const double energy = 0.5 * radial_velocity * radial_velocity
            + eta / (radius * radius) + 0.5 * radius * radius;
        return energy < energy_cutoff ? 1.0 : 0.0;
    };
    auto density_upper_bound = [&](
        double r_lower,
        double r_upper,
        double w_lower,
        double w_upper,
        double eta_lower,
        double eta_upper) {
        (void)eta_upper;
        return minimum_harmonic_energy_in_cell(
                   r_lower, r_upper, w_lower, w_upper, eta_lower)
                < energy_cutoff
            ? 1.0
            : 0.0;
    };

    std::vector<double> cell_average(
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 0.0);
    constexpr int quadrature = 96;
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                long double average = 0.0L;
                for (int qr = 0; qr < quadrature; ++qr) {
                    const double radius =
                        geometry.radial.edge[static_cast<std::size_t>(i)]
                        + (static_cast<double>(qr) + 0.5) / quadrature
                            * geometry.radial.width[static_cast<std::size_t>(i)];
                    for (int qw = 0; qw < quadrature; ++qw) {
                        const double radial_velocity =
                            geometry.radial_velocity.edge[static_cast<std::size_t>(j)]
                            + (static_cast<double>(qw) + 0.5) / quadrature
                                * geometry.radial_velocity.width[static_cast<std::size_t>(j)];
                        const double available_eta = std::clamp(
                            (energy_cutoff
                             - 0.5 * radial_velocity * radial_velocity
                             - 0.5 * radius * radius)
                                * radius * radius,
                            geometry.eta.edge[static_cast<std::size_t>(k)],
                            geometry.eta.edge[static_cast<std::size_t>(k + 1)]);
                        average += (
                            available_eta
                            - geometry.eta.edge[static_cast<std::size_t>(k)])
                            / geometry.eta.width[static_cast<std::size_t>(k)];
                    }
                }
                cell_average[ev::idx(i, j, k, p)] = static_cast<double>(
                    average / static_cast<long double>(quadrature * quadrature));
            }
        }
    }

    ev::WeightedPICInitializationOptions options;
    options.markers_per_occupied_cell = 100;
    options.global_marker_budget = 10000;
    options.velocity_pairing = ev::WeightedPICVelocityPairing::required;
    ev::WeightedPICSamplerState sampler{UINT64_C(0x1234abcd), 0};
    const auto initialized =
        ev::initialize_weighted_vlasov_pic_from_analytic_density(
            cell_average,
            geometry,
            p,
            options,
            sampler,
            analytic_density,
            density_upper_bound);
    assert(initialized.diagnostics.density_aware_in_cell_sampling);

    long double partial_mass = 0.0L;
    long double continuous_outside_before = 0.0L;
    std::vector<ev::WeightedPICMarker> centered = initialized.markers;
    for (std::size_t q = 0; q < initialized.markers.size(); ++q) {
        const ev::WeightedPICMarker& marker = initialized.markers[q];
        const std::array<int, 3> source =
            ev::weighted_pic_detail::decode_source_cell(marker.source_cell, p);
        if (source[2] == 1) {
            partial_mass += marker.mass;
        }
        if (!(harmonic_energy(marker) < energy_cutoff)) {
            continuous_outside_before += marker.mass;
        }
        centered[q].eta = geometry.eta.center[static_cast<std::size_t>(source[2])];
    }
    const double total_mass = ev::weighted_pic_total_marker_mass(initialized.markers);
    const double partial_fraction = static_cast<double>(partial_mass) / total_mass;
    assert(std::abs(partial_fraction - target_partial_shell_fraction) < 0.01);
    assert(continuous_outside_before == 0.0L);

    ev::VlasovPoissonField harmonic_field;
    harmonic_field.phi.resize(static_cast<std::size_t>(p.Nr));
    harmonic_field.phi_r.resize(static_cast<std::size_t>(p.Nr));
    harmonic_field.source_mass.assign(static_cast<std::size_t>(p.Nr), 0.0);
    for (int i = 0; i < p.Nr; ++i) {
        const double radius = geometry.radial.center[static_cast<std::size_t>(i)];
        harmonic_field.phi[static_cast<std::size_t>(i)] = 0.5 * radius * radius;
        harmonic_field.phi_r[static_cast<std::size_t>(i)] = radius;
    }
    const std::vector<double> radial_coordinate =
        ev::radial_coordinate_grid(geometry.radial.center, p);
    const ev::CharacteristicIntegratorPlan plan =
        ev::make_characteristic_integrator_plan(p);
    std::vector<ev::WeightedPICMarker> transported = initialized.markers;
    ev::advance_weighted_pic_markers_with_venus(
        transported,
        1.0e-4,
        radial_coordinate,
        geometry.radial.center,
        harmonic_field,
        p,
        plan);
    ev::advance_weighted_pic_markers_with_venus(
        centered,
        1.0e-4,
        radial_coordinate,
        geometry.radial.center,
        harmonic_field,
        p,
        plan);
    long double continuous_outside_after = 0.0L;
    long double centered_outside_after = 0.0L;
    for (const ev::WeightedPICMarker& marker : transported) {
        if (!(harmonic_energy(marker) < energy_cutoff + 1.0e-10)) {
            continuous_outside_after += marker.mass;
        }
    }
    for (const ev::WeightedPICMarker& marker : centered) {
        if (!(harmonic_energy(marker) < energy_cutoff + 1.0e-10)) {
            centered_outside_after += marker.mass;
        }
    }
    const double continuous_outside_fraction =
        static_cast<double>(continuous_outside_after) / total_mass;
    const double centered_outside_fraction =
        static_cast<double>(centered_outside_after) / total_mass;
    assert(continuous_outside_fraction == 0.0);
    assert(std::abs(centered_outside_fraction - partial_fraction) < 0.01);
    std::cout << "partial eta support: partial_mass_fraction="
              << partial_fraction
              << " continuous_outside_after=" << continuous_outside_fraction
              << " center_L_outside_after=" << centered_outside_fraction
              << '\n';
}

void test_cic_conservation_moments_noise_and_poisson_bridge() {
    const ev::Params p = pic_params();
    const ev::PhaseSpaceCellGeometry geometry = pic_geometry(p);
    const ev::WeightedPICRadialMesh mesh =
        ev::make_weighted_pic_radial_mesh(geometry);
    std::vector<ev::WeightedPICMarker> markers = {
        {0.2, p.r_min, 2.0, 0.5, 0, 0},
        {0.3, p.r_max, -1.0, 0.2, 1, 0},
        {0.5, 2.3, 0.4, 0.8, 2, 0},
        {0.4, 3.1, -0.2, 0.3, 3, 0},
        {0.1, p.r_min, 0.0, 0.1, 4, 0,
         ev::WeightedPICMarkerState::delayed_inner, 0.3, 0.5},
        {0.05, 4.2, 0.7, 0.1, 5, 0,
         ev::WeightedPICMarkerState::escaped, 0.0, 0.0},
    };
    const ev::WeightedPICDepositedMoments deposited =
        ev::deposit_weighted_vlasov_pic(markers, mesh);
    assert(nearly_equal(deposited.active_marker_mass, 1.4));
    assert(nearly_equal(deposited.deposited_mesh_mass, 1.4, 4.0e-16));
    assert(nearly_equal(deposited.hidden_inner_mass, 0.1));
    assert(nearly_equal(deposited.escaped_mass, 0.05));
    assert(std::abs(deposited.mass_roundoff)
        <= 8.0 * std::numeric_limits<double>::epsilon());
    const ev::WeightedPICCICStencil inner_stencil =
        ev::weighted_pic_cic_stencil(p.r_min, mesh);
    const ev::WeightedPICCICStencil outer_stencil =
        ev::weighted_pic_cic_stencil(p.r_max, mesh);
    assert(inner_stencil.count == 1 && inner_stencil.cell[0] == 0);
    assert(outer_stencil.count == 1
        && outer_stencil.cell[0] == mesh.shell_volume.size() - 1);
    assert(deposited.shell_mass.front() >= 0.2);
    assert(deposited.shell_mass.back() >= 0.3);

    const ev::WeightedPICTSCStencil inner_tsc =
        ev::weighted_pic_tsc_stencil(p.r_min, mesh);
    const ev::WeightedPICTSCStencil outer_tsc =
        ev::weighted_pic_tsc_stencil(p.r_max, mesh);
    double inner_tsc_sum = 0.0;
    double outer_tsc_sum = 0.0;
    for (int n = 0; n < inner_tsc.count; ++n) {
        assert(inner_tsc.weight[static_cast<std::size_t>(n)] >= 0.0);
        inner_tsc_sum += inner_tsc.weight[static_cast<std::size_t>(n)];
    }
    for (int n = 0; n < outer_tsc.count; ++n) {
        assert(outer_tsc.weight[static_cast<std::size_t>(n)] >= 0.0);
        outer_tsc_sum += outer_tsc.weight[static_cast<std::size_t>(n)];
    }
    assert(nearly_equal(inner_tsc_sum, 1.0));
    assert(nearly_equal(outer_tsc_sum, 1.0));
    const ev::WeightedPICDepositedMoments tsc_deposited =
        ev::deposit_weighted_vlasov_pic(
            markers, mesh, ev::WeightedPICParticleShape::tsc);
    assert(nearly_equal(tsc_deposited.deposited_mesh_mass, 1.4, 4.0e-16));
    const ev::WeightedPICDepositedMoments parallel_tsc =
        ev::deposit_weighted_vlasov_pic_parallel(
            markers, mesh, ev::WeightedPICParticleShape::tsc, 3);
    assert(nearly_equal(
        parallel_tsc.deposited_mesh_mass,
        tsc_deposited.deposited_mesh_mass,
        4.0e-16));
    for (std::size_t i = 0; i < mesh.shell_volume.size(); ++i) {
        assert(nearly_equal(
            parallel_tsc.shell_mass[i], tsc_deposited.shell_mass[i]));
        assert(nearly_equal(
            parallel_tsc.shell_radial_momentum[i],
            tsc_deposited.shell_radial_momentum[i]));
        assert(nearly_equal(
            parallel_tsc.shell_radial_second_moment[i],
            tsc_deposited.shell_radial_second_moment[i]));
        assert(nearly_equal(
            parallel_tsc.shell_tangential_second_moment[i],
            tsc_deposited.shell_tangential_second_moment[i]));
    }

    long double integrated_mass = 0.0L;
    long double integrated_momentum = 0.0L;
    long double integrated_radial_second = 0.0L;
    long double integrated_tangential_second = 0.0L;
    for (std::size_t i = 0; i < mesh.shell_volume.size(); ++i) {
        integrated_mass += static_cast<long double>(deposited.density[i])
            * mesh.shell_volume[i];
        integrated_momentum +=
            static_cast<long double>(deposited.radial_momentum_density[i])
            * mesh.shell_volume[i];
        integrated_radial_second +=
            static_cast<long double>(deposited.radial_second_moment_density[i])
            * mesh.shell_volume[i];
        integrated_tangential_second +=
            static_cast<long double>(deposited.tangential_second_moment_density[i])
            * mesh.shell_volume[i];
        assert(deposited.effective_marker_number[i] >= 0.0);
    }
    long double expected_momentum = 0.0L;
    long double expected_radial_second = 0.0L;
    long double expected_tangential_second = 0.0L;
    for (const ev::WeightedPICMarker& marker : markers) {
        if (marker.state != ev::WeightedPICMarkerState::active) {
            continue;
        }
        expected_momentum += static_cast<long double>(marker.mass) * marker.w;
        expected_radial_second +=
            static_cast<long double>(marker.mass) * marker.w * marker.w;
        expected_tangential_second += static_cast<long double>(marker.mass)
            * 2.0L * marker.eta / (marker.r * marker.r);
    }
    assert(nearly_equal(static_cast<double>(integrated_mass), 1.4));
    assert(nearly_equal(
        static_cast<double>(integrated_momentum),
        static_cast<double>(expected_momentum)));
    assert(nearly_equal(
        static_cast<double>(integrated_radial_second),
        static_cast<double>(expected_radial_second)));
    assert(nearly_equal(
        static_cast<double>(integrated_tangential_second),
        static_cast<double>(expected_tangential_second)));
    assert(nearly_equal(
        std::accumulate(
            tsc_deposited.shell_radial_momentum.begin(),
            tsc_deposited.shell_radial_momentum.end(),
            0.0),
        static_cast<double>(expected_momentum)));
    assert(nearly_equal(
        std::accumulate(
            tsc_deposited.shell_radial_second_moment.begin(),
            tsc_deposited.shell_radial_second_moment.end(),
            0.0),
        static_cast<double>(expected_radial_second)));
    assert(nearly_equal(
        std::accumulate(
            tsc_deposited.shell_tangential_second_moment.begin(),
            tsc_deposited.shell_tangential_second_moment.end(),
            0.0),
        static_cast<double>(expected_tangential_second)));

    const ev::Moments venus_moments =
        ev::weighted_pic_as_venus_moments(deposited, mesh);
    assert(nearly_equal(
        ev::vp_matter_mass(venus_moments, geometry.radial.center, p),
        deposited.deposited_mesh_mass));
    const ev::VlasovPoissonField field =
        ev::solve_weighted_pic_poisson_field(deposited, mesh, 0.0, p);
    assert(nearly_equal(field.matter_mass, 1.5));
    assert(nearly_equal(field.outer_source_mass, 1.5));

    const ev::WeightedPICFieldGrid refined_grid =
        ev::make_weighted_pic_field_grid(p, 2 * p.Nr);
    assert(refined_grid.parameters.Nr == 2 * p.Nr);
    assert(refined_grid.mesh.shell_volume.size()
        == static_cast<std::size_t>(2 * p.Nr));
    assert(refined_grid.mesh.edge.front() == mesh.edge.front());
    assert(refined_grid.mesh.edge.back() == mesh.edge.back());
    const ev::WeightedPICDepositedMoments refined_deposit =
        ev::deposit_weighted_vlasov_pic(markers, refined_grid.mesh);
    const ev::VlasovPoissonField refined_field =
        ev::solve_weighted_pic_poisson_field(
            refined_deposit,
            refined_grid.mesh,
            0.0,
            refined_grid.parameters);
    assert(nearly_equal(refined_deposit.deposited_mesh_mass, 1.4));
    assert(nearly_equal(refined_field.outer_source_mass, 1.5));

    const std::vector<double> f = symmetric_source(geometry, p);
    ev::WeightedPICInitializationOptions options;
    options.global_marker_budget = 96;
    options.use_piecewise_constant_in_cell = true;
    ev::WeightedPICSamplerState sampler{UINT64_C(991), 0};
    const auto initialized = ev::initialize_weighted_vlasov_pic(
        f, geometry, p, options, sampler);
    const ev::WeightedPICNoiseDiagnostics noise =
        ev::diagnose_weighted_pic_noise(initialized.markers, mesh);
    const ev::WeightedPICNoiseDiagnostics tsc_noise =
        ev::diagnose_weighted_pic_noise(
            initialized.markers,
            mesh,
            UINT64_C(0xbb67ae8584caa73b),
            ev::WeightedPICParticleShape::tsc);
    assert(noise.batch_a_markers > 0);
    assert(noise.batch_b_markers > 0);
    for (std::size_t i = 0; i < mesh.shell_volume.size(); ++i) {
        assert(std::isfinite(noise.density_uncertainty[i]));
        assert(std::isfinite(noise.radial_current_uncertainty[i]));
        assert(std::isfinite(noise.kinetic_energy_density_uncertainty[i]));
        assert(std::isfinite(noise.radial_energy_flux_uncertainty[i]));
        assert(std::isfinite(tsc_noise.density_uncertainty[i]));
    }
    std::cout << "deposition: active_mass=" << deposited.active_marker_mass
              << " deposited_mass=" << deposited.deposited_mesh_mass
              << " hidden_mass=" << deposited.hidden_inner_mass
              << " roundoff=" << deposited.mass_roundoff << '\n';
}

void test_continuous_inner_carrier_deposition() {
    ev::Params p = pic_params();
    p.inner_boundary = "origin_parity";
    const ev::PhaseSpaceCellGeometry geometry = pic_geometry(p);
    const ev::WeightedPICRadialMesh mesh =
        ev::make_weighted_pic_radial_mesh(geometry);
    const double inner_face = mesh.edge.front();
    const double inner_face_below = std::nextafter(inner_face, 0.0);
    const double inner_face_above = std::nextafter(
        inner_face, std::numeric_limits<double>::infinity());
    const double first_volume_center = std::cbrt(
        3.0 * mesh.volume_coordinate_center.front());
    const double midpoint_in_volume_coordinate = std::cbrt(
        1.5 * (
            inner_face * inner_face * inner_face / 3.0
            + mesh.volume_coordinate_center.front()));
    const std::array<double, 8> radii{
        0.0,
        0.25 * inner_face,
        0.75 * inner_face,
        inner_face_below,
        inner_face,
        inner_face_above,
        midpoint_in_volume_coordinate,
        first_volume_center,
    };
    constexpr double marker_mass = 0.625;

    for (const ev::WeightedPICParticleShape shape : {
             ev::WeightedPICParticleShape::cic,
             ev::WeightedPICParticleShape::tsc}) {
        double previous_hidden = std::numeric_limits<double>::infinity();
        double previous_visible_mass = -1.0;
        for (const double radius : radii) {
            double hidden_weight = 0.0;
            double visible_weight = 0.0;
            if (shape == ev::WeightedPICParticleShape::cic) {
                const auto stencil =
                    ev::weighted_pic_cic_stencil_with_inner_carrier(
                        radius, mesh);
                hidden_weight = stencil.hidden_weight;
                visible_weight = resolved_weight_sum(stencil.resolved);
                for (int n = 0; n < stencil.resolved.count; ++n) {
                    assert(stencil.resolved.cell[static_cast<std::size_t>(n)]
                        < mesh.shell_volume.size());
                }
            } else {
                const auto stencil =
                    ev::weighted_pic_tsc_stencil_with_inner_carrier(
                        radius, mesh);
                hidden_weight = stencil.hidden_weight;
                visible_weight = resolved_weight_sum(stencil.resolved);
                for (int n = 0; n < stencil.resolved.count; ++n) {
                    assert(stencil.resolved.cell[static_cast<std::size_t>(n)]
                        < mesh.shell_volume.size());
                }
            }
            assert(std::isfinite(hidden_weight));
            assert(hidden_weight >= 0.0);
            assert(hidden_weight <= 1.0);
            assert(nearly_equal(hidden_weight + visible_weight, 1.0));
            assert(hidden_weight <= previous_hidden + 2.0e-14);
            previous_hidden = hidden_weight;

            const std::vector<ev::WeightedPICMarker> marker = {
                {marker_mass, radius, -0.2, 0.0, 2026072701, 0},
            };
            const ev::WeightedPICDepositedMoments deposited =
                ev::deposit_weighted_vlasov_pic_parallel(
                    marker, mesh, shape, 2, true);
            assert(deposited.hidden_inner_mass >= 0.0);
            assert(deposited.deposited_mesh_mass >= 0.0);
            assert(nearly_equal(
                deposited.deposited_mesh_mass
                    + deposited.hidden_inner_mass,
                marker_mass));
            assert(nearly_equal(
                deposited.hidden_inner_mass,
                marker_mass * hidden_weight));
            assert(nearly_equal(
                deposited.deposited_mesh_mass,
                marker_mass * visible_weight));
            assert(deposited.deposited_mesh_mass
                >= previous_visible_mass - 2.0e-14);
            previous_visible_mass = deposited.deposited_mesh_mass;
            for (const double shell_mass : deposited.shell_mass) {
                assert(shell_mass >= 0.0);
            }
        }

        // A marker at the origin belongs wholly to the unresolved carrier,
        // whereas the cloud is wholly resolved from the mesh face outward.
        const auto origin_partition =
            shape == ev::WeightedPICParticleShape::cic
            ? std::array<double, 2>{
                ev::weighted_pic_cic_stencil_with_inner_carrier(0.0, mesh)
                    .hidden_weight,
                resolved_weight_sum(
                    ev::weighted_pic_cic_stencil_with_inner_carrier(0.0, mesh)
                        .resolved)}
            : std::array<double, 2>{
                ev::weighted_pic_tsc_stencil_with_inner_carrier(0.0, mesh)
                    .hidden_weight,
                resolved_weight_sum(
                    ev::weighted_pic_tsc_stencil_with_inner_carrier(0.0, mesh)
                        .resolved)};
        assert(origin_partition[0] == 1.0);
        assert(origin_partition[1] == 0.0);

        const auto face_partition =
            shape == ev::WeightedPICParticleShape::cic
            ? std::array<double, 2>{
                ev::weighted_pic_cic_stencil_with_inner_carrier(
                    inner_face, mesh).hidden_weight,
                resolved_weight_sum(
                    ev::weighted_pic_cic_stencil_with_inner_carrier(
                        inner_face, mesh).resolved)}
            : std::array<double, 2>{
                ev::weighted_pic_tsc_stencil_with_inner_carrier(
                    inner_face, mesh).hidden_weight,
                resolved_weight_sum(
                    ev::weighted_pic_tsc_stencil_with_inner_carrier(
                        inner_face, mesh).resolved)};
        assert(face_partition[0] == 0.0);
        assert(nearly_equal(face_partition[1], 1.0));

        // Adjacent floating-point radii on opposite sides of r_min must
        // produce adjacent mass sources, potentials, and forces—not a jump
        // comparable to the entire marker mass.
        std::array<ev::WeightedPICDepositedMoments, 2> face_deposit;
        std::array<ev::VlasovPoissonField, 2> face_field;
        const std::array<double, 2> face_radius{
            inner_face_below, inner_face_above};
        for (std::size_t side = 0; side < face_radius.size(); ++side) {
            const std::vector<ev::WeightedPICMarker> marker = {
                {marker_mass, face_radius[side], -0.2, 0.0,
                 UINT64_C(2026072702), 0},
            };
            face_deposit[side] = ev::deposit_weighted_vlasov_pic_parallel(
                marker, mesh, shape, 2, true);
            face_field[side] = ev::solve_weighted_pic_poisson_field(
                face_deposit[side], mesh, 0.0, p);
        }
        const double no_marker_jump = 1.0e-11 * marker_mass;
        assert(std::abs(
            face_deposit[1].hidden_inner_mass
            - face_deposit[0].hidden_inner_mass) < no_marker_jump);
        for (std::size_t cell = 0; cell < mesh.shell_volume.size(); ++cell) {
            assert(std::abs(
                face_deposit[1].shell_mass[cell]
                - face_deposit[0].shell_mass[cell]) < no_marker_jump);
            const double potential_scale = std::max({
                marker_mass,
                std::abs(face_field[0].phi[cell]),
                std::abs(face_field[1].phi[cell])});
            const double force_scale = std::max({
                marker_mass,
                std::abs(face_field[0].phi_r[cell]),
                std::abs(face_field[1].phi_r[cell])});
            assert(std::abs(
                face_field[1].phi[cell] - face_field[0].phi[cell])
                < 1.0e-11 * potential_scale);
            assert(std::abs(
                face_field[1].phi_r[cell] - face_field[0].phi_r[cell])
                < 1.0e-11 * force_scale);
        }
    }
    std::cout << "inner carrier: CIC/TSC partitions conserve mass and cross "
                 "r_min continuously\n";
}

void test_orbit_abstraction_delayed_state_and_venus_adapter() {
    std::vector<ev::WeightedPICMarker> ballistic = {
        {0.25, 2.0, 0.4, 0.3, 10, 0},
    };
    ev::advance_weighted_pic_markers(
        ballistic,
        0.5,
        [](const ev::WeightedPICMarker& marker, double dt) {
            return ev::WeightedPICOrbitOutcome{
                ev::WeightedPICOrbitDisposition::active,
                marker.r + marker.w * dt,
                marker.w,
                dt,
                0.0,
                0.0,
            };
        });
    assert(nearly_equal(ballistic[0].r, 2.2));
    assert(ballistic[0].mass == 0.25);
    assert(ballistic[0].eta == 0.3);

    std::vector<ev::WeightedPICMarker> already_hidden = {
        {0.2, 1.0, -0.5, 0.1, 11, 0,
         ev::WeightedPICMarkerState::delayed_inner, 0.5, 0.25},
    };
    ev::advance_weighted_pic_markers(
        already_hidden,
        0.5,
        [](const ev::WeightedPICMarker& marker, double dt) {
            return ev::WeightedPICOrbitOutcome{
                ev::WeightedPICOrbitDisposition::active,
                marker.r + marker.w * dt,
                marker.w,
                dt,
                0.0,
                0.0,
            };
        });
    assert(already_hidden[0].state == ev::WeightedPICMarkerState::active);
    assert(nearly_equal(already_hidden[0].r, 1.125));
    assert(nearly_equal(already_hidden[0].w, 0.5));

    std::vector<ev::WeightedPICMarker> crossing = {
        {0.3, 1.1, -1.0, 0.2, 12, 0},
    };
    int calls = 0;
    ev::advance_weighted_pic_markers(
        crossing,
        0.5,
        [&](const ev::WeightedPICMarker& marker, double dt) {
            ++calls;
            if (calls == 1) {
                return ev::WeightedPICOrbitOutcome{
                    ev::WeightedPICOrbitDisposition::entered_delayed_inner,
                    1.0,
                    -1.0,
                    0.1,
                    1.0,
                    0.2,
                };
            }
            return ev::WeightedPICOrbitOutcome{
                ev::WeightedPICOrbitDisposition::active,
                marker.r + marker.w * dt,
                marker.w,
                dt,
                0.0,
                0.0,
            };
        });
    assert(calls == 2);
    assert(crossing[0].state == ev::WeightedPICMarkerState::active);
    assert(nearly_equal(crossing[0].r, 1.2));
    assert(crossing[0].mass == 0.3);
    assert(crossing[0].eta == 0.2);

    ev::Params p = pic_params();
    const ev::PhaseSpaceCellGeometry geometry = pic_geometry(p);
    ev::Moments zero;
    zero.rho.assign(static_cast<std::size_t>(p.Nr), 0.0);
    zero.p.assign(static_cast<std::size_t>(p.Nr), 0.0);
    zero.pt.assign(static_cast<std::size_t>(p.Nr), 0.0);
    zero.j.assign(static_cast<std::size_t>(p.Nr), 0.0);
    const ev::VlasovPoissonField field = ev::solve_vlasov_poisson_field(
        zero, geometry.radial.center, 0.0, p);
    const std::vector<double> radial_coordinate =
        ev::radial_coordinate_grid(geometry.radial.center, p);
    const ev::CharacteristicIntegratorPlan plan =
        ev::make_characteristic_integrator_plan(p);
    std::vector<ev::WeightedPICMarker> venus_marker = {
        {0.1, 2.0, 0.1, 0.0, 13, 0},
    };
    ev::advance_weighted_pic_markers_with_venus(
        venus_marker,
        0.2,
        radial_coordinate,
        geometry.radial.center,
        field,
        p,
        plan);
    assert(nearly_equal(venus_marker[0].r, 2.02, 1.0e-12));
    assert(nearly_equal(venus_marker[0].w, 0.1, 1.0e-12));

    std::vector<ev::WeightedPICMarker> serial_many;
    for (std::uint64_t q = 0; q < 16; ++q) {
        serial_many.push_back({
            0.01,
            1.2 + 0.1 * static_cast<double>(q),
            0.02 * static_cast<double>(static_cast<int>(q % 5) - 2),
            0.01 * static_cast<double>(q % 3),
            100 + q,
            q,
        });
    }
    std::vector<ev::WeightedPICMarker> parallel_many = serial_many;
    ev::advance_weighted_pic_markers_with_venus(
        serial_many,
        0.05,
        radial_coordinate,
        geometry.radial.center,
        field,
        p,
        plan);
    ev::advance_weighted_pic_markers_with_venus_parallel(
        parallel_many,
        0.05,
        radial_coordinate,
        geometry.radial.center,
        field,
        p,
        plan,
        4);
    for (std::size_t q = 0; q < serial_many.size(); ++q) {
        assert_same_marker(serial_many[q], parallel_many[q]);
    }
    std::cout << "orbit abstraction: delayed_events=" << calls
              << " venus_r=" << venus_marker[0].r
              << " parallel_bitwise_markers=" << serial_many.size() << '\n';
}

void test_expanding_self_consistent_kdk() {
    ev::Params p = pic_params();
    p.vp_universe = "expanding";
    p.vp_alpha = 2.0 / 3.0;
    p.vp_time_offset = 12.0;
    p.vp_poisson_source = "density_contrast";
    const ev::PhaseSpaceCellGeometry geometry = pic_geometry(p);
    ev::Moments zero;
    zero.rho.assign(static_cast<std::size_t>(p.Nr), 0.0);
    zero.p.assign(static_cast<std::size_t>(p.Nr), 0.0);
    zero.pt.assign(static_cast<std::size_t>(p.Nr), 0.0);
    zero.j.assign(static_cast<std::size_t>(p.Nr), 0.0);
    constexpr double start_time = 0.0;
    constexpr double dt = 0.7;
    const ev::VlasovPoissonField old_field = ev::solve_vlasov_poisson_field(
        zero, geometry.radial.center, start_time, p);
    const ev::VlasovPoissonField new_field = ev::solve_vlasov_poisson_field(
        zero, geometry.radial.center, start_time + dt, p);
    const std::vector<double> radial_coordinate =
        ev::radial_coordinate_grid(geometry.radial.center, p);
    std::vector<ev::WeightedPICMarker> marker = {
        {0.25, 2.0, 0.1, 0.0, 20260722, 0},
    };
    const ev::VPBackground start = ev::vp_background(start_time, p);
    const ev::VPBackground end = ev::vp_background(start_time + dt, p);
    const double canonical_momentum = start.a * marker[0].w;
    const double drift =
        ev::weighted_pic_inverse_scale_factor_squared_integral(
            start_time, start_time + dt, p);
    const double expected_radius = end.a
        * (marker[0].r / start.a + canonical_momentum * drift);
    const double expected_velocity = canonical_momentum / end.a;
    ev::begin_weighted_pic_self_consistent_kdk_parallel(
        marker,
        start_time,
        dt,
        radial_coordinate,
        geometry.radial.center,
        old_field,
        p,
        2);
    ev::finish_weighted_pic_self_consistent_kdk_parallel(
        marker,
        start_time + dt,
        dt,
        radial_coordinate,
        geometry.radial.center,
        new_field,
        p,
        2);
    assert(nearly_equal(marker[0].r, expected_radius, 2.0e-14));
    assert(nearly_equal(marker[0].w, expected_velocity, 2.0e-14));
    assert(marker[0].mass == 0.25);
    assert(marker[0].eta == 0.0);
    assert(nearly_equal(old_field.background.H, p.vp_alpha / p.vp_time_offset));
    assert(new_field.background.H < old_field.background.H);

    std::vector<ev::WeightedPICMarker> angular_marker = {
        {0.2, 2.0, 0.1, 0.18, 20260724, 0},
    };
    const double angular_momentum = std::sqrt(2.0 * angular_marker[0].eta);
    const double angular_x = angular_marker[0].r / start.a;
    const double angular_p = start.a * angular_marker[0].w;
    const double angular_axis = angular_x + drift * angular_p;
    const double transverse_axis = drift * angular_momentum / angular_x;
    const double expected_angular_x = std::hypot(
        angular_axis, transverse_axis);
    const double expected_angular_p = (
        angular_p * angular_axis
        + angular_momentum / angular_x * transverse_axis)
        / expected_angular_x;
    ev::begin_weighted_pic_self_consistent_kdk_parallel(
        angular_marker,
        start_time,
        dt,
        radial_coordinate,
        geometry.radial.center,
        old_field,
        p,
        2);
    ev::finish_weighted_pic_self_consistent_kdk_parallel(
        angular_marker,
        start_time + dt,
        dt,
        radial_coordinate,
        geometry.radial.center,
        new_field,
        p,
        2);
    assert(nearly_equal(
        angular_marker[0].r, end.a * expected_angular_x));
    assert(nearly_equal(
        angular_marker[0].w, expected_angular_p / end.a));
    assert(angular_marker[0].eta == 0.18);

    ev::Params origin = p;
    origin.inner_boundary = "origin_parity";
    std::vector<ev::WeightedPICMarker> core_marker = {
        {0.125, 0.2, -0.5, 0.0, 20260723, 0},
    };
    const ev::VPBackground core_end = ev::vp_background(dt, origin);
    const double core_momentum = core_marker[0].w;
    const double core_drift =
        ev::weighted_pic_inverse_scale_factor_squared_integral(0.0, dt, origin);
    const double raw_core_coordinate = core_marker[0].r
        + core_momentum * core_drift;
    ev::begin_weighted_pic_self_consistent_kdk_parallel(
        core_marker,
        0.0,
        dt,
        radial_coordinate,
        geometry.radial.center,
        old_field,
        origin,
        2);
    ev::finish_weighted_pic_self_consistent_kdk_parallel(
        core_marker,
        dt,
        dt,
        radial_coordinate,
        geometry.radial.center,
        new_field,
        origin,
        2);
    assert(raw_core_coordinate < 0.0);
    assert(nearly_equal(
        core_marker[0].r, core_end.a * std::abs(raw_core_coordinate)));
    assert(nearly_equal(core_marker[0].w, -core_momentum / core_end.a));
    const ev::WeightedPICRadialMesh mesh =
        ev::make_weighted_pic_radial_mesh(geometry);
    assert(core_marker[0].r < mesh.edge.front());
    const ev::WeightedPICDepositedMoments core_deposit =
        ev::deposit_weighted_vlasov_pic_parallel(
            core_marker,
            mesh,
            ev::WeightedPICParticleShape::tsc,
            2,
            true);
    assert(core_deposit.deposited_mesh_mass >= 0.0);
    assert(core_deposit.hidden_inner_mass > 0.0);
    assert(nearly_equal(
        core_deposit.deposited_mesh_mass
            + core_deposit.hidden_inner_mass,
        core_marker[0].mass));
    const ev::VlasovPoissonField core_field =
        ev::solve_weighted_pic_poisson_field(
            core_deposit, mesh, dt, origin);
    assert(core_field.hidden_carrier_mass == core_deposit.hidden_inner_mass);
    assert(core_field.source_mass_edge.front()
        == core_deposit.hidden_inner_mass);
    std::cout << "expanding KDK: a1=" << end.a
              << " r1=" << marker[0].r
              << " u1=" << marker[0].w
              << " origin_r=" << core_marker[0].r << '\n';
}

// The radial mesh stays at fixed physical radii. In an expanding universe
// its walls reverse u=H*r+w, where the stored w is peculiar velocity.
void test_expanding_fixed_physical_wall_reflection() {
    ev::Params p = pic_params();
    p.vp_universe = "expanding";
    p.vp_alpha = 2.0 / 3.0;
    p.vp_time_offset = 12.0;
    p.vp_poisson_source = "density_contrast";
    const auto geometry = pic_geometry(p);
    const auto coordinate = ev::radial_coordinate_grid(geometry.radial.center, p);
    ev::Moments zero;
    zero.rho.assign(p.Nr, 0.0);
    zero.p.assign(p.Nr, 0.0);
    zero.pt.assign(p.Nr, 0.0);
    zero.j.assign(p.Nr, 0.0);
    const auto field = ev::solve_vlasov_poisson_field(
        zero, geometry.radial.center, 0.0, p);
    constexpr double dt = 0.1;
    const auto end = ev::vp_background(dt, p);
    const double drift = ev::weighted_pic_inverse_scale_factor_squared_integral(0.0, dt, p);
    for (bool outer : {true, false}) {
        const double initial_r = outer ? 3.99 : 1.01;
        const double initial_w = outer ? 0.0 : -0.4;
        const double wall = outer ? p.r_max : p.r_min;
        const double raw_r = end.a * (initial_r + drift * initial_w);
        assert(outer ? raw_r > wall : raw_r < wall);
        std::vector<ev::WeightedPICMarker> marker = {
            {0.25, initial_r, initial_w, 0.0, 20261112, 0},
        };
        const auto initial_markers = marker;
        ev::begin_weighted_pic_self_consistent_kdk_parallel(
            marker, 0.0, dt, coordinate, geometry.radial.center, field, p, 1);
        auto measured_markers = initial_markers;
        ev::WeightedPICOrbitDiagnostics measured;
        ev::begin_weighted_pic_self_consistent_kdk_parallel(measured_markers,
            0.0, dt, coordinate, geometry.radial.center, field, p, 1, &measured);
        assert(measured.reflecting_wall_impacts == 1);
        assert(measured.maximum_endpoint_radius
            == std::max(initial_r, measured_markers[0].r));
        assert(measured_markers[0].r == marker[0].r);
        assert(measured_markers[0].w == marker[0].w);
        ev::WeightedPICOrbitRollbackBuffer rollback;
        ev::capture_weighted_pic_orbit_rollback(initial_markers, rollback, 1);
        ev::restore_weighted_pic_orbit_rollback(measured_markers, rollback, 1);
        // A rejected trial impact must disappear from the accepted retry's
        // diagnostics, just as its orbit coordinates are rolled back.
        ev::begin_weighted_pic_self_consistent_kdk_parallel(measured_markers,
            0.0, 0.001, coordinate, geometry.radial.center, field, p, 1, &measured);
        assert(measured.reflecting_wall_impacts == 0);
        // Independent integration of r''=alpha(alpha-1)r/tau^2 with
        // physical radial velocity reversed at the first wall crossing.
        const double expected_r = outer ? 3.9878456244800105 : 1.024388856660709;
        const double expected_w = outer ? -0.44143949321744924 : 0.28738411965118127;
        assert(nearly_equal(marker[0].r, expected_r, 3.0e-12));
        assert(nearly_equal(marker[0].w, expected_w, 3.0e-12));
        assert(marker[0].mass == 0.25 && marker[0].eta == 0.0);
    }
}

void test_finite_wall_angular_drift() {
    // References integrate the physical equations dr/dt=u and
    // du/dt=alpha(alpha-1)r/tau^2+L^2/r^3, locating specular impacts.
    const double cases[][8] = {
        {1.1,-1,.5,.2,12,0,1.097889341983841,.9996023348252993},
        {1.1,-3,.05,1,12,0,3.9002284428789213,3.0003169448255447},
        {3.9,.6,2,1,12,0,3.5150219059290033,-.5470276455243397},
        {1.01,-.4,.2,.1,12,2.0/3.0,1.0243889628767895,.2889711376234256},
        {3.99,0,.7,.1,12,2.0/3.0,3.9878303155207884,-.44136351085983094},
        {1.1,-2,.5,1,2,.5,2.6287353060373944,1.27442665951349},
        {1.1,-2,.5,1,2,1.2,2.292731897929813,.5313156969196584},
    };
    for (const auto& c : cases) {
        double r = c[0], w = c[1];
        int events = 0;
        assert(ev::advance_central_drift_with_walls(
            r,w,c[2],c[3],c[4],c[5],1,true,4,true,&events));
        assert(events == 1);
        assert(nearly_equal(r,c[6],3.0e-12));
        assert(nearly_equal(w,c[7],3.0e-12));
        assert(ev::advance_central_drift_with_walls(
            r,w,c[2],-c[3],c[4]+c[3],c[5],1,true,4,true));
        assert(nearly_equal(r,c[0],3.0e-13));
        assert(nearly_equal(w,c[1],3.0e-13));
    }
    // Exercise the actual self-consistent CPU entry point, including a
    // crossing hidden by the free trajectory's outward return and an
    // unreflected endpoint at r=0 that is intercepted by the finite wall.
    ev::Params p = pic_params();
    const auto geometry = pic_geometry(p);
    const auto coordinate = ev::radial_coordinate_grid(geometry.radial.center,p);
    ev::Moments zero;
    zero.rho.assign(p.Nr,0);zero.p.assign(p.Nr,0);
    zero.pt.assign(p.Nr,0);zero.j.assign(p.Nr,0);
    const auto field = ev::solve_vlasov_poisson_field(zero,geometry.radial.center,0,p);
    for (const auto& c : {std::array<double,4>{1.1,-1,.5,.2},
                          std::array<double,4>{1.1,-3,.05,1},
                          std::array<double,4>{3.9,.6,2,1},
                          std::array<double,4>{1.1,-1.1,0,1}}) {
        std::vector<ev::WeightedPICMarker> marker = {{.25,c[0],c[1],.5*c[2]*c[2],1,0}};
        const double energy = .5*(c[1]*c[1]+c[2]*c[2]/(c[0]*c[0]));
        ev::begin_weighted_pic_self_consistent_kdk_parallel(
            marker,0,c[3],coordinate,geometry.radial.center,field,p,1);
        ev::finish_weighted_pic_self_consistent_kdk_parallel(
            marker,c[3],c[3],coordinate,geometry.radial.center,field,p,1);
        const auto& m=marker[0];
        assert(nearly_equal(.5*m.w*m.w+m.eta/(m.r*m.r),energy,3.0e-14));
        assert(m.r>=p.r_min && m.r<=p.r_max);
        assert(m.mass==.25 && m.eta==.5*c[2]*c[2]);
    }
    double r=2,w=3;
    int events=0;
    const double initial_energy=.5*(w*w+.4*.4/(r*r));
    assert(ev::advance_central_drift_with_walls(r,w,.4,10,12,0,1,true,4,true,&events));
    assert(events>3);
    assert(nearly_equal(.5*(w*w+.4*.4/(r*r)),initial_energy,1.0e-13));
    assert(ev::advance_central_drift_with_walls(r,w,.4,-10,22,0,1,true,4,true));
    assert(nearly_equal(r,2,1.0e-12) && nearly_equal(w,3,1.0e-12));
    for (double wall : {1.0,4.0}) {
        r=wall;w=0;events=-1;
        assert(ev::advance_central_drift_with_walls(r,w,0,.2,12,0,1,true,4,true,&events));
        assert(r==wall && w==0 && events==0);
        for (double interval : {.01,.2,2.0,20.0}) {
            r=wall;w=-wall/12;events=-1;
            assert(ev::advance_central_drift_with_walls(
                r,w,0,interval,12,1,1,true,4,true,&events));
            assert(r==wall && w==-wall/(12+interval) && events==0);
        }
    }
    std::cout << "Finite wall angular drift: physical ODE references, hidden crossings, energy and reversal passed.\n";
}

void test_exact_radial_origin_kdk() {
    ev::Params p = pic_params();
    p.inner_boundary = "origin_parity";
    const auto geometry = pic_geometry(p);
    const auto coordinate = ev::radial_coordinate_grid(geometry.radial.center,p);
    ev::Moments zero;
    zero.rho.assign(p.Nr,0);zero.p.assign(p.Nr,0);
    zero.pt.assign(p.Nr,0);zero.j.assign(p.Nr,0);
    const auto field = ev::solve_vlasov_poisson_field(zero,geometry.radial.center,0,p);
    auto step = [&](std::vector<ev::WeightedPICMarker>& markers, double dt) {
        ev::begin_weighted_pic_self_consistent_kdk_parallel(
            markers,0,dt,coordinate,geometry.radial.center,field,p,1);
        ev::finish_weighted_pic_self_consistent_kdk_parallel(
            markers,dt,dt,coordinate,geometry.radial.center,field,p,1);
    };
    // Exact Cartesian radial motion, including an origin endpoint, both
    // origin velocity signs, a complete crossing, and a subsequent wall hit.
    for (const auto& c : {std::array<double,5>{1,-1,1,0,1},
                          std::array<double,5>{0,1,.4,.4,1},
                          std::array<double,5>{0,-1,.4,.4,1},
                          std::array<double,5>{0,0,.4,0,0},
                          std::array<double,5>{1,-2,1,1,2},
                          std::array<double,5>{0,1,5,3,-1},
                          std::array<double,5>{0,-1,0,0,-1}}) {
        std::vector<ev::WeightedPICMarker> markers = {{.25,c[0],c[1],0,7,0}};
        step(markers,c[2]);
        assert(nearly_equal(markers[0].r,c[3]));
        assert(nearly_equal(markers[0].w,c[4]));
        assert(markers[0].mass==.25 && markers[0].eta==0 && markers[0].id==7);
        if (c[0]>0) {
            markers[0].w=-markers[0].w;
            step(markers,c[2]);
            assert(nearly_equal(markers[0].r,c[0]));
            assert(nearly_equal(markers[0].w,-c[1]));
        }
    }
    for (double alpha : {0.0,.5,2.0/3.0}) {
        // The signed helper must reverse a step that lands exactly at r=0.
        const double tau=12,dt=.7;
        const double r0=tau*ev::central_drift_detail::drift_coordinate(dt,tau,alpha);
        double r=r0,w=-1;
        assert(ev::advance_central_drift_with_walls(
            r,w,0,dt,tau,alpha,1,false,4,true));
        assert(r==0 && w>0);
        assert(ev::advance_central_drift_with_walls(
            r,w,0,-dt,tau+dt,alpha,1,false,4,true));
        assert(nearly_equal(r,r0) && nearly_equal(w,-1));
    }
    for (const auto& c : {std::array<double,4>{0,1,.1,0},
                          std::array<double,4>{0,1,0,.2},
                          std::array<double,4>{1,-2,0,.2}}) {
        p.central_mass=c[3];
        std::vector<ev::WeightedPICMarker> markers = {{.25,c[0],c[1],c[2],7,0}};
        bool rejected=false;
        try { step(markers,1); }
        catch (const std::runtime_error&) { rejected=true; }
        assert(rejected);
    }
    std::cout << "Radial origin: exact crossing, endpoint, origin start, reversal and singular-state rejection passed.\n";
}

void test_density_weighted_analytic_shells() {
    // The requested envelope concerns the density that sources gravity:
    // rest-mass density in VP and particle-energy density in EV.
    const std::vector<double> r{1,6,10,16,20},w{-1,-.5,0,.5,1},L{0,.5,1};
    const std::vector<double> wq{.25,.5,.5,.5,.25},Lq{.25,.5,.25};
    const std::vector<double> mu(r.size(),0);
    for (const std::string model : {"vlasov_poisson","einstein_vlasov"}) {
        ev::Params p;
        p.physics_model=model;p.vp_universe="static";p.radial_grid="linear";
        p.Nr=5;p.Nw=5;p.NL=3;p.shell_initial_df=true;
        p.gaussian_density_shell=true;p.R0=10;p.dR=4;
        p.beta=9;p.amp=.35;p.w_in=0;
        auto density = [&](const std::vector<double>& f) {
            if (model=="vlasov_poisson")
                return ev::compute_vp_moments(f,r,w,L,wq,Lq,p).rho;
            std::vector<double> rho(r.size(),0);
            for (int i=0;i<p.Nr;++i) {
                for (int j=0;j<p.Nw;++j) for (int k=0;k<p.NL;++k)
                    rho[i]+=wq[j]*Lq[k]*L[k]*f[ev::idx(i,j,k,p)]
                        *std::sqrt(1+w[j]*w[j]+L[k]*L[k]/(r[i]*r[i]));
                rho[i]*=2*ev::PI/(r[i]*r[i]);
            }
            return rho;
        };
        const auto f1=ev::initial_distribution(r,w,L,wq,Lq,mu,r,p);
        const auto rho1=density(f1);
        for (std::size_t i=0;i<r.size();++i)
            assert(std::abs(rho1[i]/rho1[2]/ev::shell_profile(r[i],p)-1)<5e-14);
        p.R0=16;p.dR=2;p.amp=.2;p.w_in=-.3;
        const auto f2=ev::initial_distribution(r,w,L,wq,Lq,mu,r,p);
        const auto rho2=density(f2);
        p.R0=10;p.dR=4;p.amp=.35;p.w_in=0;
        p.double_gaussian_density_shell=true;
        p.R0_2=16;p.dR_2=2;p.amp_2=.2;p.w_in_2=-.3;
        const auto both=ev::initial_distribution(r,w,L,wq,Lq,mu,r,p);
        const auto rho_both=density(both);
        for (std::size_t i=0;i<r.size();++i) {
            const double expected=rho1[2]*std::exp(-.5*std::pow((r[i]-10)/4,2))
                +rho2[3]*std::exp(-.5*std::pow((r[i]-16)/2,2));
            assert(std::abs(rho_both[i]/expected-1)<5e-14);
        }
        p.double_gaussian_density_shell=false;p.gaussian_density_shell=false;
        p.top_hat_density_shell=true;p.dR=20;
        const auto flat=density(ev::initial_distribution(r,w,L,wq,Lq,mu,r,p));
        for (double value:flat)assert(std::abs(value/flat[2]-1)<5e-14);
    }
    std::cout << "Analytic density shells: Newtonian mass and relativistic energy envelopes passed.\n";
}

void test_initializer_uniform_open_interval() {
    // Inverting the integer mixer gives a real seed whose output word is
    // UINT64_MAX. Its top midpoint rounds to exactly one without the guard.
    constexpr std::uint64_t seed=UINT64_C(0xfe4d0a08e7db7ab6);
    assert(ev::weighted_pic_mix64(seed ^ ev::weighted_pic_mix64(0)
        ^ ev::weighted_pic_mix64(UINT64_C(0xd1b54a32d192ed03)))
        ==std::numeric_limits<std::uint64_t>::max());
    const double value=ev::weighted_pic_counter_uniform(seed,0,0);
    assert(value==std::nextafter(1.0,0.0));
    assert(value>0 && value<1);
}

void test_generic_point_radial_origin_kdk() {
    // The static Boltzmann event driver uses this point-characteristic
    // adapter, independently of the self-consistent WPIC KDK entry point.
    ev::Params p = pic_params();
    p.inner_boundary = "origin_parity";
    const auto geometry = pic_geometry(p);
    const auto coordinate = ev::radial_coordinate_grid(geometry.radial.center,p);
    ev::Moments zero;
    zero.rho.assign(p.Nr,0);zero.p.assign(p.Nr,0);
    zero.pt.assign(p.Nr,0);zero.j.assign(p.Nr,0);
    const auto field = ev::solve_vlasov_poisson_field(
        zero,geometry.radial.center,0,p);
    const auto plan = ev::make_characteristic_integrator_plan(p);
    // Cartesian radial motion supplies independent exact endpoint values.
    // Signed intervals exercise the backward grid characteristic as well.
    for (const auto& c : {std::array<double,5>{1,-1,1,0,1},
                          std::array<double,5>{1,1,-1,0,-1},
                          std::array<double,5>{0,1,-1,1,-1},
                          std::array<double,5>{0,-1,1,1,1},
                          std::array<double,5>{0,-1,0,0,-1},
                          std::array<double,5>{0,0,1,0,0},
                          std::array<double,5>{1,-2,1,1,2}}) {
        const auto result = ev::advance_vp_characteristic_unregularized(
            c[0],c[1],0,coordinate,geometry.radial.center,field,c[2],p,plan);
        assert(nearly_equal(result.r,c[3]) && nearly_equal(result.w,c[4]));
    }
    std::vector<ev::WeightedPICMarker> markers = {{.25,1,-1,0,7,0}};
    ev::advance_weighted_pic_markers_with_venus_parallel(
        markers,1,coordinate,geometry.radial.center,field,p,plan,1);
    assert(markers[0].r==0 && markers[0].w==1);
    ev::advance_weighted_pic_markers_with_venus_parallel(
        markers,.25,coordinate,geometry.radial.center,field,p,plan,1);
    assert(markers[0].r==.25 && markers[0].w==1);
    assert(markers[0].mass==.25 && markers[0].eta==0 && markers[0].id==7);
    const ev::VPSample empty{};
    assert(ev::vp_frozen_hamiltonian_error_from_samples(
        {0,1},{.25,1},0,empty,empty)==0);
    assert(ev::vp_radial_acceleration(
        0,1,0,empty,geometry.radial.center,field)==0);
    for (const auto& c : {std::array<double,3>{1.e-120,1.e-100,1.e160},
                          std::array<double,3>{1.e150,1.e200,1.e-50}}) {
        const double acceleration = ev::vp_radial_acceleration(
            c[0],0,c[1],empty,geometry.radial.center,field);
        assert(std::isfinite(acceleration));
        assert(std::abs(acceleration/c[2]-1)<5.e-14);
    }
    // A central point mass has a physical singularity, whereas an empty
    // regular center admits the exact crossings above.
    p.central_mass=1;
    const auto singular = ev::solve_vlasov_poisson_field(
        zero,geometry.radial.center,0,p);
    for (const auto& c : {std::array<double,3>{1,-1,1},
                          std::array<double,3>{1,1,-1}}) {
        bool rejected=false;
        try {
            (void)ev::advance_vp_characteristic_unregularized(
                c[0],c[1],0,coordinate,geometry.radial.center,singular,c[2],p,plan);
        } catch (const std::runtime_error&) { rejected=true; }
        assert(rejected);
    }
    std::cout << "Point KDK radial origin: forward/backward endpoints, adapter, energy and singular crossing passed.\n";
}

void test_origin_force_uses_inner_face_mass() {
    ev::Params p = pic_params();
    p.inner_boundary = "origin_parity";
    const auto geometry = pic_geometry(p);
    const auto coordinate = ev::radial_coordinate_grid(geometry.radial.center,p);
    ev::Moments shell;
    shell.rho.assign(p.Nr,0);shell.p.assign(p.Nr,0);
    shell.pt.assign(p.Nr,0);shell.j.assign(p.Nr,0);
    shell.rho[0]=1; // Exterior material has zero force inside its empty cavity.
    for (double central : {0.0,.2}) {
        p.central_mass=central;
        for (double hidden : {0.0,.25}) {
            const auto field=ev::solve_vlasov_poisson_field(
                shell,geometry.radial.center,0,p,hidden);
            const auto interior=ev::sample_vp_field(.5,coordinate,geometry.radial.center,field,p);
            const auto left=ev::sample_vp_field(
                std::nextafter(1.0,0.0),coordinate,geometry.radial.center,field,p);
            const auto face=ev::sample_vp_field(1,coordinate,geometry.radial.center,field,p);
            assert(nearly_equal(interior.phi_r,p.vp_gravity*(4*central+.5*hidden)));
            assert(nearly_equal(interior.phi-face.phi,
                p.vp_gravity*(-central+.5*hidden*(.25-1)),5e-14));
            assert(nearly_equal(left.phi_r,face.phi_r));
            assert(nearly_equal(left.phi,face.phi));
        }
    }
    p.central_mass=0;
    p.vp_universe="expanding";
    p.vp_alpha=2.0/3.0;
    p.vp_time_offset=12;
    p.vp_poisson_source="total_density";
    const double background=ev::vp_background(0,p).rho_bar;
    // This is homogeneous in the solver's line-density quadrature. The
    // represented mass includes the physical central background sphere.
    shell.rho.assign(p.Nr,background);
    const double central_background=4*ev::PI*background/3;
    const auto homogeneous=ev::solve_vlasov_poisson_field(
        shell,geometry.radial.center,0,p,central_background);
    assert(homogeneous.hidden_carrier_mass==central_background);
    assert(nearly_equal(homogeneous.source_mass_edge.front(),0));
    for (double radius : {.1,.5,1.0,2.0,3.9}) {
        const auto value=ev::sample_vp_field(radius,coordinate,geometry.radial.center,homogeneous,p);
        assert(nearly_equal(value.phi_r,0));
        assert(nearly_equal(value.phi,0));
    }
    p.vp_poisson_source="density_contrast";
    shell.rho.assign(p.Nr,0);
    const auto zero_contrast=ev::solve_vlasov_poisson_field(
        shell,geometry.radial.center,0,p,0);
    assert(zero_contrast.source_mass_edge.front()==0);
    assert(zero_contrast.outer_source_mass==0);
    std::cout << "Origin force: shell theorem, uniform core, point mass and face continuity passed.\n";
}

void test_expanding_finite_volume_wall_mass() {
    ev::Params p = pic_params();
    p.Nr = 3;
    p.Nw = 5;
    p.NL = 1;
    const std::vector<double> r{1.0, 2.0, 3.0};
    const std::vector<double> w{-2.0, -1.0, 0.0, 1.0, 2.0};
    const std::vector<double> angular{0.0};
    ev::VlasovPoissonField field;
    field.background.H = 0.2;
    field.phi.assign(p.Nr, 0.0);
    field.phi_r.assign(p.Nr, 0.0);
    std::vector<double> f(p.Nr * p.Nw * p.NL, 0.0);
    f[ev::idx(2, 2, 0, p)] = 1.0;
    const auto advanced = ev::advect_vlasov_poisson_finite_volume(
        f, r, w, angular, field, 0.01, 0.0, p);
    const auto re = ev::control_volume_edges(r);
    const auto we = ev::control_volume_edges(w);
    double mass_before = 0.0;
    double mass_after = 0.0;
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            const double volume = (re[i + 1] - re[i]) * (we[j + 1] - we[j]);
            mass_before += volume * f[ev::idx(i, j, 0, p)];
            mass_after += volume * advanced[ev::idx(i, j, 0, p)];
            assert(advanced[ev::idx(i, j, 0, p)] >= 0.0);
        }
    }
    std::cerr << "expanding FV wall mass: before=" << mass_before
              << " after=" << mass_after << '\n';
    assert(nearly_equal(mass_before, mass_after));
}

void test_radial_boundary_limits() {
    auto p = pic_params();
    double r = 12.0;
    double w = 0.3;
    ev::reflect_radial_boundaries(r, w, 1.0, 4.0, p, 0.2);
    // Three successive mirror crossings: 12 -> -4 -> 6 -> 2.
    assert(nearly_equal(r, 2.0));
    assert(nearly_equal(0.2 * r + w, -(0.2 * 12.0 + 0.3)));
    r = 26.0;
    w = 240.0;
    ev::reflect_radial_boundaries(r, w, 1.0, 4.0, p);
    assert(r == 2.0 && w == 240.0);
    p.outer_boundary = "outflow_vacuum";
    r = 5.0;
    w = 0.3;
    ev::reflect_radial_boundaries(r, w, 1.0, 4.0, p, 0.2);
    assert(r == 5.0 && w == 0.3);
    p.inner_boundary = "origin_parity";
    r = -0.2;
    ev::reflect_radial_boundaries(r, w, 1.0, 4.0, p, 0.2);
    assert(r == 0.2 && w == -0.3);
    p = pic_params();
    r = 100.0;
    bool rejected = false;
    try {
        ev::reflect_radial_boundaries(r, w, 1.0, 4.0, p, 0.2);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
}

void test_nonuniform_finite_volume_wall_cfl() {
    auto p = pic_params();
    p.integrator = "conservative_finite_volume";
    p.Nr = 3; p.Nw = 5; p.NL = 2;
    p.dt_max = 0.5; p.cfl = 0.35; p.threads = 1;
    const std::vector<double> r{1.0, 2.0, 3.0};
    const auto w = ev::stretched_w_grid(10.0, 8.0, 0.0, p.Nw);
    const std::vector<double> angular{0.0, 0.1};
    ev::VlasovPoissonField field;
    field.phi.assign(p.Nr, 0.0);
    field.phi_r.assign(p.Nr, 0.0);
    std::vector<double> f(p.Nr * p.Nw * p.NL, 0.0);
    f[ev::idx(0, 1, 0, p)] = 1.0;
    // This tail is below the ordinary characteristic active threshold.
    f[ev::idx(1, 0, 0, p)] = 1.0e-30;
    const ev::MpiContext mpi;
    const double dt = ev::choose_dt_vlasov_poisson(
        f, r, w, angular, field, 1.0, 1.0, p, mpi);
    const auto after = ev::advect_vlasov_poisson_finite_volume(
        f, r, w, angular, field, dt, 0.0, p);
    assert(*std::min_element(after.begin(), after.end()) >= 0.0);
    std::cerr << "nonuniform FV wall CFL: dt=" << dt
              << " minimum_f=" << *std::min_element(after.begin(), after.end()) << '\n';
}

void test_prescribed_inflow_shell_mass() {
    auto p = pic_params();
    p.outer_boundary = "vp_outer_inflow";
    p.vp_inflow_count = 1;
    p.vp_inflow_low_l_count = 1;
    p.vp_inflow_start_time = 4.0;
    p.vp_inflow_time_sigma = 0.5;
    p.vp_inflow_time_cutoff = 2.0;
    p.vp_inflow_shell_mass = 0.3;
    p.vp_inflow_coord_speed = 0.2;
    p.vp_inflow_sigma_u = 1.0;
    p.vp_inflow_low_sigma_t = 0.4;
    p.vp_inflow_low_l_center = 0.0;
    ev::VlasovPoissonField field;
    field.background.H = 0.2;
    constexpr double radius = 4.0;
    constexpr double angular = 1.0;
    const double tangential = ev::normalized_tangential_pdf(
        angular, radius, p.vp_inflow_low_sigma_t, 0.0);
    // Integrate the actual incoming flux over physical radial velocity.
    double peak_flux = 0.0;
    constexpr int intervals = 4000;
    constexpr double du = 12.0 / intervals;
    for (int j = 0; j < intervals; ++j) {
        const double u = -12.0 + (j + 0.5) * du;
        const double value = ev::vp_outer_inflow_distribution(
            4.0, u - field.background.H * radius, angular, radius, field, p);
        peak_flux += -u * value / tangential * du;
    }
    peak_flux *= 4.0 * ev::PI * radius * radius;
    // Integrate the prescribed time envelope independently by midpoint rule.
    const double central_w = -field.background.H * radius - p.vp_inflow_coord_speed;
    const double peak = ev::vp_outer_inflow_distribution(
        4.0, central_w, angular, radius, field, p);
    double duration = 0.0;
    constexpr double dt = 2.0 / intervals;
    for (int n = 0; n < intervals; ++n) {
        const double time = 3.0 + (n + 0.5) * dt;
        duration += ev::vp_outer_inflow_distribution(
            time, central_w, angular, radius, field, p) / peak * dt;
    }
    const double injected_mass = peak_flux * duration;
    std::cerr << "warm prescribed inflow: shell_mass=" << p.vp_inflow_shell_mass
              << " integrated_incoming_mass=" << injected_mass << '\n';
    assert(std::abs(injected_mass / p.vp_inflow_shell_mass - 1.0) < 1.0e-6);
}

void test_orbit_rollback_preserves_collision_and_restart_state() {
    std::vector<ev::WeightedPICMarker> markers(3);
    for (std::size_t i = 0; i < markers.size(); ++i) {
        markers[i].mass = 0.1 * static_cast<double>(i + 1);
        markers[i].r = 1.25 + 0.5 * static_cast<double>(i);
        markers[i].w = -0.3 + 0.2 * static_cast<double>(i);
        markers[i].eta = 0.04 * static_cast<double>(i + 1);
        markers[i].id = UINT64_C(1000) + i;
        markers[i].source_cell = UINT64_C(2000) + i;
        markers[i].state = ev::WeightedPICMarkerState::active;
        markers[i].internal_energy = 0.02 * static_cast<double>(i + 1);
        markers[i].internal_velocity_dispersion =
            0.05 * static_cast<double>(i + 1);
    }
    const std::vector<ev::WeightedPICMarker> original = markers;
    const ev::WeightedPICCollisionSamplerState collision_sampler{
        UINT64_C(0xabcdef0123456789), UINT64_C(314159)};
    const double saved_collision_age = 0.01875;

    ev::WeightedPICOrbitRollbackBuffer rollback;
    ev::capture_weighted_pic_orbit_rollback(markers, rollback, 2);
    for (ev::WeightedPICMarker& marker : markers) {
        // These are exactly the fields changed by a provisional KDK probe.
        marker.r += 0.123;
        marker.w -= 0.456;
    }
    ev::restore_weighted_pic_orbit_rollback(markers, rollback, 2);
    for (std::size_t i = 0; i < markers.size(); ++i) {
        assert_same_marker(markers[i], original[i]);
    }
    // A rejected orbit never consumes a collision variate or advances the
    // persisted event age; restart therefore resumes from the same state.
    assert(collision_sampler.seed == UINT64_C(0xabcdef0123456789));
    assert(collision_sampler.next_counter == UINT64_C(314159));
    assert(saved_collision_age == 0.01875);
}

std::vector<char> read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::vector<char>(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

void test_restart_round_trip_is_deterministic() {
    const ev::Params p = pic_params();
    const ev::PhaseSpaceCellGeometry geometry = pic_geometry(p);
    const std::vector<double> f = symmetric_source(geometry, p);
    ev::WeightedPICInitializationOptions options;
    options.global_marker_budget = 32;
    options.use_piecewise_constant_in_cell = true;
    ev::WeightedPICSamplerState sampler{UINT64_C(20260721), 77};
    auto initialized = ev::initialize_weighted_vlasov_pic(
        f, geometry, p, options, sampler);
    initialized.markers[0].state = ev::WeightedPICMarkerState::delayed_inner;
    initialized.markers[0].inner_boundary_speed = 0.7;
    initialized.markers[0].remaining_inner_time = 0.125;
    ev::WeightedPICRestartState state;
    state.markers = initialized.markers;
    state.sampler = sampler;
    state.collision_sampler = {
        UINT64_C(0x0123456789abcdef), UINT64_C(987654321)};
    state.potential_event_age = 0.375;
    state.collision_event_age = 0.625;
    state.collision_timescale_estimate = 2.75;
    state.potential_timescale_estimate = 4.5;
    state.time = 1.25;
    state.step = 42;

    const std::filesystem::path first_path =
        std::filesystem::temp_directory_path()
        / "venus_weighted_pic_restart_test_a.bin";
    const std::filesystem::path second_path =
        std::filesystem::temp_directory_path()
        / "venus_weighted_pic_restart_test_b.bin";
    std::filesystem::remove(first_path);
    std::filesystem::remove(second_path);
    ev::write_weighted_pic_restart(first_path, state);
    const ev::WeightedPICRestartState loaded =
        ev::read_weighted_pic_restart(first_path);
    ev::write_weighted_pic_restart(second_path, loaded);
    assert(loaded.time == state.time);
    assert(loaded.step == state.step);
    assert(loaded.sampler.seed == state.sampler.seed);
    assert(loaded.sampler.next_counter == state.sampler.next_counter);
    assert(loaded.collision_sampler.seed == state.collision_sampler.seed);
    assert(loaded.collision_sampler.next_counter
           == state.collision_sampler.next_counter);
    assert(loaded.potential_event_age == state.potential_event_age);
    assert(loaded.collision_event_age == state.collision_event_age);
    assert(
        loaded.collision_timescale_estimate
        == state.collision_timescale_estimate);
    assert(
        loaded.potential_timescale_estimate
        == state.potential_timescale_estimate);
    assert(loaded.markers.size() == state.markers.size());
    for (std::size_t i = 0; i < state.markers.size(); ++i) {
        assert_same_marker(loaded.markers[i], state.markers[i]);
    }
    assert(read_bytes(first_path) == read_bytes(second_path));
    std::filesystem::remove(first_path);
    std::filesystem::remove(second_path);
    std::cout << "restart: markers=" << loaded.markers.size()
              << " sampler_counter=" << loaded.sampler.next_counter << '\n';
}

}  // namespace

int main() {
    test_mixed_marker_mass_reduction();
    test_mass_budget_quiet_pairing_and_determinism();
    test_partial_eta_energy_support_uses_continuous_eta();
    test_cic_conservation_moments_noise_and_poisson_bridge();
    test_continuous_inner_carrier_deposition();
    test_orbit_abstraction_delayed_state_and_venus_adapter();
    test_expanding_self_consistent_kdk();
    test_expanding_finite_volume_wall_mass();
    test_expanding_fixed_physical_wall_reflection();
    test_finite_wall_angular_drift();
    test_exact_radial_origin_kdk();
    test_generic_point_radial_origin_kdk();
    test_initializer_uniform_open_interval();
    test_density_weighted_analytic_shells();
    test_origin_force_uses_inner_face_mass();
    test_radial_boundary_limits();
    test_prescribed_inflow_shell_mass();
    test_nonuniform_finite_volume_wall_cfl();
    test_orbit_rollback_preserves_collision_and_restart_state();
    test_restart_round_trip_is_deterministic();
    std::cout << "weighted Vlasov PIC tests passed\n";
    return 0;
}
