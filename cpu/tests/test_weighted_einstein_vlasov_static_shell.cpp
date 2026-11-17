// Focused regression coverage for the neutral static Einstein--Vlasov shell
// generator and its invariant-weight interface to Schwarzschild WPIC.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ev/weighted_einstein_vlasov_pic_schwarzschild.hpp"
#include "ev/weighted_einstein_vlasov_static_shell.hpp"

namespace {

using ev::static_einstein_vlasov::MarkerSamplingOptions;
using ev::static_einstein_vlasov::MarkerSamplingRule;
using ev::static_einstein_vlasov::MarkerSet;
using ev::static_einstein_vlasov::Parameters;
using ev::static_einstein_vlasov::Solution;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    double value,
    double reference,
    double relative_tolerance,
    double absolute_tolerance,
    const std::string& message) {
    require(
        std::abs(value - reference)
            <= absolute_tolerance
                + relative_tolerance
                    * std::max(std::abs(value), std::abs(reference)),
        message + ": value=" + std::to_string(value)
            + ", reference=" + std::to_string(reference));
}

ev::Params schwarzschild_parameters(double outer_radius, int radial_cells) {
    ev::Params p;
    p.physics_model = "einstein_vlasov";
    p.vp_universe = "static";
    p.bp_collision_operator = "none";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "uniform";
    p.vp_interpolation = "hamiltonian_sldg";
    p.Nr = radial_cells;
    p.Nw = 4;
    p.NL = 2;
    p.r_min = 0.0;
    p.r_max = outer_radius;
    p.radial_grid = "linear";
    p.w_max = 1.0;
    p.L_max = 4.0;
    p.inner_boundary = "origin_parity";
    p.outer_boundary = "outflow_vacuum";
    p.metric_floor = 1.0e-12;
    p.gap_stop = 1.0e-4;
    return p;
}

void test_continuum_solution(const Solution& shell) {
    // Independent high-accuracy continuum reference for
    // C2=1, L0=1, y(0)=0.13.  These values describe the unscaled solution.
    require_close(shell.adm_mass, 0.54912, 8.0e-5, 2.0e-6,
        "static-shell ADM reference changed");
    require_close(shell.rest_weight, 0.57198, 8.0e-5, 2.0e-6,
        "static-shell rest-weight reference changed");
    require_close(shell.cutoff_energy, 0.85549, 8.0e-5, 2.0e-6,
        "static-shell cutoff-energy reference changed");
    require_close(shell.inner_matter_radius, 1.835156, 8.0e-6, 2.0e-7,
        "static-shell inner-edge reference changed");
    require_close(shell.outer_matter_radius, 3.3463, 5.0e-4, 3.0e-6,
        "static-shell outer-edge reference changed");
    require_close(shell.binding_fraction, 0.03996, 3.0e-4, 3.0e-6,
        "static-shell binding reference changed");
    require_close(shell.diagnostics.maximum_compactness, 0.3479,
        5.0e-4, 5.0e-6, "static-shell compactness reference changed");

    require(shell.cutoff_energy < 1.0,
        "static shell must obey the finite-support condition E0<1");
    require(shell.binding_energy > 0.0,
        "candidate equilibrium must have positive global binding energy");
    require(shell.diagnostics.minimum_compactness_gap > 0.60,
        "candidate equilibrium unexpectedly approaches a trapped surface");
    require(shell.diagnostics.constraint_residual_linf < 2.0e-5,
        "static radial Einstein-constraint residual is too large: "
            + std::to_string(
                shell.diagnostics.constraint_residual_linf));
    require(shell.diagnostics.jeans_residual_relative_l2 < 2.0e-5,
        "static anisotropic Jeans/TOV residual is too large: "
            + std::to_string(
                shell.diagnostics.jeans_residual_relative_l2));
    require_close(
        ev::static_einstein_vlasov::
            delta_angular_momentum_squared_amplitude(shell.parameters),
        1.0, 2.0e-14, 2.0e-14,
        "delta(L^2-L0^2) coefficient has the wrong Jacobian");

    const double log_cutoff = std::log(shell.cutoff_energy);
    for (std::size_t i = 0; i < shell.profile.size(); ++i) {
        const auto& node = shell.profile[i];
        require(node.radius >= shell.inner_matter_radius
                && node.radius <= shell.outer_matter_radius,
            "static-shell profile node left the matter interval");
        require_close(node.mu + node.y, log_cutoff,
            2.0e-13, 2.0e-14, "mu+y is not log(E0)");
        require_close(node.dm_dr,
            4.0 * ev::PI * node.radius * node.radius * node.sources.rho,
            2.0e-13, 2.0e-14, "mass constraint is inconsistent at a node");
        const double gap = 1.0 - 2.0 * node.mass / node.radius;
        const double expected_y_derivative = -(
            node.mass / (node.radius * node.radius)
            + 4.0 * ev::PI * node.radius * node.sources.radial_pressure)
            / gap;
        require_close(node.dy_dr, expected_y_derivative,
            2.0e-13, 2.0e-14, "lapse constraint is inconsistent at a node");
    }

    const auto inner_vacuum =
        ev::static_einstein_vlasov::evaluate_profile(shell, 0.5);
    require(inner_vacuum.mass == 0.0
            && inner_vacuum.sources.rho == 0.0,
        "regular inner vacuum was not extended exactly");
    const double exterior_radius = 2.0 * shell.outer_matter_radius;
    const auto exterior =
        ev::static_einstein_vlasov::evaluate_profile(
            shell, exterior_radius);
    const double exterior_gap =
        1.0 - 2.0 * shell.adm_mass / exterior_radius;
    require_close(exterior.mass, shell.adm_mass, 0.0, 0.0,
        "Schwarzschild exterior changed the ADM mass");
    require_close(exterior.mu, 0.5 * std::log(exterior_gap),
        2.0e-14, 2.0e-15,
        "Schwarzschild exterior lapse normalization is wrong");
    require_close(exterior.lambda, -exterior.mu,
        2.0e-14, 2.0e-15,
        "Schwarzschild exterior does not obey lambda+mu=0");
}

Solution test_shooting_and_rescaling(const Solution& shell) {
    Parameters shooting = shell.parameters;
    const Solution recovered =
        ev::static_einstein_vlasov::shoot_to_cutoff_energy(
            shooting, shell.cutoff_energy, 0.12, 0.14, 2.0e-9, 60);
    require_close(recovered.parameters.central_y, shell.parameters.central_y,
        2.0e-7, 2.0e-9, "cutoff-energy shooting did not recover y(0)");
    require_close(recovered.cutoff_energy, shell.cutoff_energy,
        3.0e-9, 3.0e-12, "cutoff-energy shooting missed its target");

    const Solution scaled =
        ev::static_einstein_vlasov::rescale_to_adm_mass(shell, 0.85);
    const double scale = 0.85 / shell.adm_mass;
    require_close(scaled.adm_mass, 0.85, 2.0e-14, 2.0e-14,
        "static-shell ADM scaling missed its target");
    require_close(scaled.parameters.angular_momentum,
        scale * shell.parameters.angular_momentum, 2.0e-14, 2.0e-14,
        "static-shell angular momentum did not scale with radius");
    require_close(scaled.inner_matter_radius,
        scale * shell.inner_matter_radius, 2.0e-14, 2.0e-14,
        "static-shell inner radius did not scale");
    require_close(scaled.outer_matter_radius,
        scale * shell.outer_matter_radius, 2.0e-14, 2.0e-14,
        "static-shell outer radius did not scale");
    require_close(scaled.cutoff_energy, shell.cutoff_energy,
        0.0, 0.0, "dimensionless cutoff energy changed under scaling");
    require_close(scaled.binding_fraction, shell.binding_fraction,
        0.0, 0.0, "binding fraction changed under scaling");
    require_close(scaled.diagnostics.maximum_compactness,
        shell.diagnostics.maximum_compactness, 0.0, 0.0,
        "compactness changed under scaling");
    require_close(
        ev::static_einstein_vlasov::
            delta_angular_momentum_squared_amplitude(scaled.parameters),
        1.0, 2.0e-14, 2.0e-14,
        "C2 changed under the static scaling symmetry");
    return scaled;
}

MarkerSet test_marker_sampling(const Solution& shell) {
    MarkerSamplingOptions coarse_options;
    coarse_options.radial_bins = 64;
    coarse_options.radial_quadrature_order = 2;
    coarse_options.positive_momentum_quadrature_order = 4;
    const MarkerSet coarse =
        ev::static_einstein_vlasov::sample_markers(shell, coarse_options);

    MarkerSamplingOptions fine_options;
    fine_options.radial_bins = 512;
    fine_options.radial_quadrature_order = 4;
    fine_options.positive_momentum_quadrature_order = 8;
    fine_options.first_marker_id = 17;
    const MarkerSet fine =
        ev::static_einstein_vlasov::sample_markers(shell, fine_options);
    require(fine.rest_weight_relative_error < coarse.rest_weight_relative_error,
        "invariant rest-weight quadrature did not converge");
    require(fine.rest_weight_relative_error < 2.0e-6,
        "fine marker quadrature does not reproduce continuum rest weight");
    require(fine.maximum_energy_over_cutoff < 1.0,
        "a sampled marker violates E<E0");
    require(fine.markers.size() % 2 == 0 && !fine.markers.empty(),
        "static-shell marker set is not paired");

    long double global_current = 0.0L;
    for (std::size_t i = 0; i < fine.markers.size(); i += 2) {
        const auto& positive = fine.markers[i];
        const auto& negative = fine.markers[i + 1];
        require(positive.mass == negative.mass
                && positive.r == negative.r
                && positive.eta == negative.eta
                && positive.w == -negative.w,
            "adjacent deterministic momentum pair is not exactly antisymmetric");
        require(positive.id + 1 == negative.id,
            "static-shell marker IDs are not deterministic and consecutive");
        require(positive.state == ev::WeightedPICMarkerState::active
                && negative.state == ev::WeightedPICMarkerState::active,
            "static-shell marker was not emitted active");
        const long double radius = positive.r;
        const long double epsilon = std::sqrt(
            1.0L + static_cast<long double>(positive.w) * positive.w
            + 2.0L * positive.eta / (radius * radius));
        global_current += positive.mass * positive.w / epsilon;
        global_current += negative.mass * negative.w / epsilon;
    }
    require(global_current == 0.0L,
        "paired marker current does not cancel in extended precision");
    return fine;
}

std::uint64_t marker_phase_space_fingerprint(const MarkerSet& sampled) {
    std::uint64_t fingerprint = UINT64_C(1469598103934665603);
    auto mix = [&](double value) {
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        fingerprint ^= bits;
        fingerprint *= UINT64_C(1099511628211);
    };
    for (const auto& marker : sampled.markers) {
        mix(marker.mass);
        mix(marker.r);
        mix(marker.w);
        mix(marker.eta);
    }
    return fingerprint;
}

void require_exact_momentum_pairs(const MarkerSet& sampled) {
    require(sampled.markers.size() % 2 == 0 && !sampled.markers.empty(),
        "shifted static-shell marker set is not paired");
    long double current = 0.0L;
    for (std::size_t i = 0; i < sampled.markers.size(); i += 2) {
        const auto& positive = sampled.markers[i];
        const auto& negative = sampled.markers[i + 1];
        require(positive.mass == negative.mass
                && positive.r == negative.r
                && positive.eta == negative.eta
                && positive.w == -negative.w,
            "shifted realization lost exact adjacent +/-w pairing");
        current += static_cast<long double>(positive.mass) * positive.w;
        current += static_cast<long double>(negative.mass) * negative.w;
    }
    require(current == 0.0L,
        "shifted realization has nonzero paired canonical current");
}

void test_independent_shifted_realizations(const Solution& shell) {
    auto sample = [&](std::uint64_t realization, std::size_t radial_bins) {
        MarkerSamplingOptions options;
        options.radial_bins = radial_bins;
        options.radial_quadrature_order = 4;
        options.positive_momentum_quadrature_order = 8;
        options.rule = MarkerSamplingRule::shifted_stratified;
        options.realization = realization;
        return ev::static_einstein_vlasov::sample_markers(shell, options);
    };
    const MarkerSet fine0 = sample(0, 512);
    const MarkerSet fine1 = sample(1, 512);
    require_exact_momentum_pairs(fine0);
    require_exact_momentum_pairs(fine1);
    require(marker_phase_space_fingerprint(fine0)
            != marker_phase_space_fingerprint(fine1),
        "realization IDs 0 and 1 produced identical phase-space fingerprints");
    require(fine0.rest_weight_relative_error < 8.0e-4
            && fine1.rest_weight_relative_error < 8.0e-4,
        "shifted-stratified realization has excessive quadrature error");
    require(fine0.maximum_energy_over_cutoff < 1.0
            && fine1.maximum_energy_over_cutoff < 1.0,
        "shifted-stratified realization sampled outside E<E0");

    // CP-shifted lattice rules are unbiased realization ensembles, but the
    // rank-one generator changes with N, so same-seed point sets are not
    // nested.  Test convergence statistically across enough independent
    // shifts instead of imposing a mathematically unjustified per-seed
    // monotonicity assertion on realizations 0 and 1.
    constexpr std::uint64_t realization_count = 16;
    std::vector<double> coarse_error;
    std::vector<double> fine_error;
    coarse_error.reserve(realization_count);
    fine_error.reserve(realization_count);
    long double coarse_square = 0.0L;
    long double fine_square = 0.0L;
    for (std::uint64_t realization = 0;
         realization < realization_count;
         ++realization) {
        const double coarse =
            sample(realization, 64).rest_weight_relative_error;
        const double fine =
            sample(realization, 512).rest_weight_relative_error;
        coarse_error.push_back(coarse);
        fine_error.push_back(fine);
        coarse_square += coarse * coarse;
        fine_square += fine * fine;
    }
    const double coarse_rms = std::sqrt(
        static_cast<double>(coarse_square / realization_count));
    const double fine_rms = std::sqrt(
        static_cast<double>(fine_square / realization_count));
    std::sort(coarse_error.begin(), coarse_error.end());
    std::sort(fine_error.begin(), fine_error.end());
    const std::size_t p90_index = static_cast<std::size_t>(
        std::ceil(0.90 * realization_count)) - 1;
    const double coarse_p90 = coarse_error[p90_index];
    const double fine_p90 = fine_error[p90_index];
    require(fine_rms < 0.65 * coarse_rms
            && fine_p90 < 0.75 * coarse_p90,
        "CP-shifted lattice ensemble did not converge in RMS and p90");
    std::cout << std::setprecision(16)
        << "static_shell_shifted_realizations K=" << realization_count
        << " error0=" << fine0.rest_weight_relative_error
        << " error1=" << fine1.rest_weight_relative_error
        << " RMS_64=" << coarse_rms
        << " RMS_512=" << fine_rms
        << " p90_64=" << coarse_p90
        << " p90_512=" << fine_p90
        << " fingerprint0=" << marker_phase_space_fingerprint(fine0)
        << " fingerprint1=" << marker_phase_space_fingerprint(fine1)
        << '\n';
}

struct DepositionDiagnostics {
    double adm_relative_error = 0.0;
    double maximum_current = 0.0;
    double maximum_lambda_t = 0.0;
    double source_relative_l2 = 0.0;
    double mass_profile_relative_l2 = 0.0;
    double lambda_profile_relative_l2 = 0.0;
    double mu_profile_relative_l2 = 0.0;
};

DepositionDiagnostics deposit_and_compare(
    const Solution& shell,
    const MarkerSet& sampled,
    int radial_cells) {
    const double field_outer_radius = 1.45 * shell.outer_matter_radius;
    const ev::Params p =
        schwarzschild_parameters(field_outer_radius, radial_cells);
    const ev::WeightedPICFieldGrid grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(p, radial_cells);
    const auto deposited =
        ev::deposit_weighted_einstein_vlasov_pic_parallel(
            sampled.markers, grid.mesh, ev::WeightedPICParticleShape::tsc, 1);
    const auto field = ev::solve_weighted_einstein_vlasov_pic_field(
        deposited, grid);

    require_close(deposited.active_rest_mass,
        sampled.represented_rest_weight, 3.0e-13, 3.0e-14,
        "WPIC deposit changed invariant marker rest weight");
    require(std::abs(deposited.rest_mass_roundoff)
            < 2.0e-13 * sampled.represented_rest_weight,
        "WPIC rest-weight deposition roundoff is excessive");

    DepositionDiagnostics result;
    result.adm_relative_error =
        std::abs(field.adm_mass - shell.adm_mass) / shell.adm_mass;
    long double source_error = 0.0L;
    long double source_reference = 0.0L;
    long double mass_error = 0.0L;
    long double mass_reference = 0.0L;
    long double lambda_error = 0.0L;
    long double lambda_reference = 0.0L;
    long double mu_error = 0.0L;
    long double mu_reference = 0.0L;
    for (std::size_t i = 0; i < grid.radius.size(); ++i) {
        result.maximum_current = std::max(
            result.maximum_current, std::abs(field.moments.j[i]));
        result.maximum_lambda_t = std::max(
            result.maximum_lambda_t, std::abs(field.metric.lambda_t[i]));
        const auto exact = ev::static_einstein_vlasov::evaluate_profile(
            shell, grid.radius[i]);
        const long double volume = grid.mesh.shell_volume[i];
        const long double density_difference =
            field.moments.rho[i] - exact.sources.rho;
        const long double pressure_difference =
            field.moments.p[i] - exact.sources.radial_pressure;
        const long double tangential_difference =
            field.moments.pt[i] - exact.sources.tangential_pressure_sum;
        source_error += volume * (
            density_difference * density_difference
            + pressure_difference * pressure_difference
            + tangential_difference * tangential_difference);
        source_reference += volume * (
            exact.sources.rho * exact.sources.rho
            + exact.sources.radial_pressure * exact.sources.radial_pressure
            + exact.sources.tangential_pressure_sum
                * exact.sources.tangential_pressure_sum);
        const long double mass_difference =
            field.metric.m[i] - exact.mass;
        const long double lambda_difference =
            field.metric.lambda[i] - exact.lambda;
        const long double mu_difference =
            field.metric.mu[i] - exact.mu;
        mass_error += volume * mass_difference * mass_difference;
        mass_reference += volume * exact.mass * exact.mass;
        lambda_error += volume * lambda_difference * lambda_difference;
        lambda_reference += volume * exact.lambda * exact.lambda;
        mu_error += volume * mu_difference * mu_difference;
        mu_reference += volume * exact.mu * exact.mu;
    }
    result.source_relative_l2 = source_reference > 0.0L
        ? std::sqrt(static_cast<double>(source_error / source_reference))
        : std::numeric_limits<double>::infinity();
    result.mass_profile_relative_l2 = mass_reference > 0.0L
        ? std::sqrt(static_cast<double>(mass_error / mass_reference))
        : std::numeric_limits<double>::infinity();
    result.lambda_profile_relative_l2 = lambda_reference > 0.0L
        ? std::sqrt(static_cast<double>(lambda_error / lambda_reference))
        : std::numeric_limits<double>::infinity();
    result.mu_profile_relative_l2 = mu_reference > 0.0L
        ? std::sqrt(static_cast<double>(mu_error / mu_reference))
        : std::numeric_limits<double>::infinity();
    return result;
}

void test_schwarzschild_deposition(
    const Solution& shell,
    const MarkerSet& sampled) {
    const DepositionDiagnostics coarse =
        deposit_and_compare(shell, sampled, 256);
    const DepositionDiagnostics fine =
        deposit_and_compare(shell, sampled, 512);
    require(fine.adm_relative_error < coarse.adm_relative_error,
        "WPIC ADM reconstruction did not converge with radial resolution");
    require(fine.source_relative_l2 < coarse.source_relative_l2,
        "WPIC source reconstruction did not converge with radial resolution");
    require(fine.mass_profile_relative_l2
                < coarse.mass_profile_relative_l2
            && fine.lambda_profile_relative_l2
                < coarse.lambda_profile_relative_l2
            && fine.mu_profile_relative_l2
                < coarse.mu_profile_relative_l2,
        "WPIC m/lambda/mu profiles did not converge with radial resolution");
    require(fine.adm_relative_error < 2.5e-3,
        "WPIC ADM reconstruction is inconsistent with the static continuum");
    require(fine.source_relative_l2 < 3.0e-2,
        "WPIC matter moments are inconsistent with the static continuum");
    require(fine.mass_profile_relative_l2 < 2.0e-3
            && fine.lambda_profile_relative_l2 < 2.0e-3
            && fine.mu_profile_relative_l2 < 2.0e-3,
        "WPIC metric profiles are inconsistent with the static continuum");
    require(fine.maximum_current < 2.0e-14
            && fine.maximum_lambda_t < 2.0e-14,
        "paired static markers do not deposit a current-free slice");

    std::cout << std::setprecision(16)
        << "static_shell M=" << shell.adm_mass
        << " M0=" << shell.rest_weight
        << " E0=" << shell.cutoff_energy
        << " Rin=" << shell.inner_matter_radius
        << " Rout=" << shell.outer_matter_radius
        << " binding_fraction=" << shell.binding_fraction
        << " Cmax=" << shell.diagnostics.maximum_compactness
        << " constraint=" << shell.diagnostics.constraint_residual_linf
        << " jeans=" << shell.diagnostics.jeans_residual_relative_l2
        << " marker_rest_error=" << sampled.rest_weight_relative_error
        << " ADM_error_256=" << coarse.adm_relative_error
        << " ADM_error_512=" << fine.adm_relative_error
        << " source_L2_256=" << coarse.source_relative_l2
        << " source_L2_512=" << fine.source_relative_l2
        << " m_L2_256=" << coarse.mass_profile_relative_l2
        << " m_L2_512=" << fine.mass_profile_relative_l2
        << " lambda_L2_256=" << coarse.lambda_profile_relative_l2
        << " lambda_L2_512=" << fine.lambda_profile_relative_l2
        << " mu_L2_256=" << coarse.mu_profile_relative_l2
        << " mu_L2_512=" << fine.mu_profile_relative_l2
        << " max_current_512=" << fine.maximum_current
        << '\n';
}

double weighted_radius_quantile(
    const std::vector<ev::WeightedPICMarker>& markers,
    double fraction) {
    require(fraction > 0.0 && fraction < 1.0,
        "invalid marker radius quantile");
    std::vector<std::pair<double, double>> radius_weight;
    radius_weight.reserve(markers.size());
    long double total = 0.0L;
    for (const auto& marker : markers) {
        if (marker.state != ev::WeightedPICMarkerState::active) {
            continue;
        }
        radius_weight.emplace_back(marker.r, marker.mass);
        total += marker.mass;
    }
    require(total > 0.0L && !radius_weight.empty(),
        "radius quantile has no active marker weight");
    std::sort(radius_weight.begin(), radius_weight.end());
    const long double target = fraction * total;
    long double cumulative = 0.0L;
    for (const auto& item : radius_weight) {
        cumulative += item.second;
        if (cumulative >= target) {
            return item.first;
        }
    }
    return radius_weight.back().first;
}

void test_short_self_consistent_equilibrium_evolution(
    const Solution& shell,
    const MarkerSet& sampled) {
    const ev::Params p = schwarzschild_parameters(
        1.45 * shell.outer_matter_radius, 512);
    const ev::WeightedPICFieldGrid grid =
        ev::make_weighted_einstein_vlasov_pic_field_grid(p, p.Nr);
    std::vector<ev::WeightedPICMarker> markers = sampled.markers;
    const auto initial_field =
        ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
            markers, grid, ev::WeightedPICParticleShape::tsc, 1);
    const double initial_r50 = weighted_radius_quantile(markers, 0.5);
    const double initial_r90 = weighted_radius_quantile(markers, 0.9);
    const double initial_rest_weight = initial_field.rest_mass;

    ev::WeightedEinsteinVlasovPICOptions options;
    options.shape = ev::WeightedPICParticleShape::tsc;
    options.threads = 1;
    options.characteristic_substeps = 2;
    constexpr double dt = 0.005;
    constexpr int steps = 8;
    ev::WeightedEinsteinVlasovPICField final_field = initial_field;
    for (int step = 0; step < steps; ++step) {
        const auto advanced =
            ev::advance_weighted_einstein_vlasov_pic_self_consistent(
                markers, dt, grid, options);
        require_close(advanced.accepted_dt, dt, 0.0, 0.0,
            "static-shell short evolution bisected a safe timestep");
        require(advanced.escaped_rest_mass == 0.0,
            "static-shell short evolution lost a marker through the boundary");
        require_close(advanced.rest_mass_after, initial_rest_weight,
            3.0e-13, 3.0e-14,
            "static-shell short evolution changed invariant rest weight");
        require(!advanced.horizon_threshold_reached,
            "static equilibrium reached a spurious horizon threshold");
        final_field = advanced.endpoint_field;
    }

    const double final_r50 = weighted_radius_quantile(markers, 0.5);
    const double final_r90 = weighted_radius_quantile(markers, 0.9);
    const double r50_drift =
        std::abs(final_r50 - initial_r50) / initial_r50;
    const double r90_drift =
        std::abs(final_r90 - initial_r90) / initial_r90;
    const double compactness_drift = std::abs(
        final_field.maximum_compactness
        - initial_field.maximum_compactness);
    require(r50_drift < 2.0e-3 && r90_drift < 2.0e-3,
        "static equilibrium has excessive short-time radius drift");
    require(compactness_drift < 2.0e-3,
        "static equilibrium has excessive short-time compactness drift");
    std::cout << std::setprecision(16)
        << "static_shell_short_evolution t=" << dt * steps
        << " r50_drift=" << r50_drift
        << " r90_drift=" << r90_drift
        << " compactness_drift=" << compactness_drift
        << " ADM_drift="
        << std::abs(final_field.adm_mass - initial_field.adm_mass)
            / initial_field.adm_mass
        << '\n';
}

}  // namespace

int main() {
    try {
        Parameters parameters;
        parameters.maximum_radius = 12.0;
        parameters.maximum_radial_step = 1.0e-2;
        const Solution unscaled =
            ev::static_einstein_vlasov::solve(parameters);
        test_continuum_solution(unscaled);
        const Solution scaled = test_shooting_and_rescaling(unscaled);
        const MarkerSet sampled = test_marker_sampling(scaled);
        test_independent_shifted_realizations(scaled);
        test_schwarzschild_deposition(scaled, sampled);
        test_short_self_consistent_equilibrium_evolution(scaled, sampled);
    } catch (const std::exception& error) {
        std::cerr << "test_weighted_einstein_vlasov_static_shell: "
                  << error.what() << '\n';
        return 1;
    }
    std::cout << "weighted Einstein-Vlasov static-shell tests passed\n";
    return 0;
}
