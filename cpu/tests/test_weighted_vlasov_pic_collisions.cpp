// Equal- and unequal-mass WPIC elastic-collision tests for NTC rates, shell
// locality, Yukawa/Plummer angular physics, determinism, and pair invariants.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "ev/weighted_vlasov_pic_collisions.hpp"

namespace {

ev::WeightedPICRadialMesh one_shell_mesh() {
    ev::WeightedPICRadialMesh mesh;
    mesh.edge = {0.5, 1.5};
    const double q_inner = std::pow(mesh.edge[0], 3) / 3.0;
    const double q_outer = std::pow(mesh.edge[1], 3) / 3.0;
    const double q = 0.5 * (q_inner + q_outer);
    mesh.volume_coordinate_center = {q};
    mesh.venus_center = {std::cbrt(3.0 * q)};
    mesh.shell_volume = {
        4.0 * ev::PI * (q_outer - q_inner),
    };
    return mesh;
}

double gaussian(ev::DSMCSplitMix64& rng) {
    return std::sqrt(-2.0 * std::log(rng.uniform_open()))
        * std::cos(2.0 * ev::PI * rng.uniform_open());
}

std::vector<ev::WeightedPICMarker> maxwellian_markers(
    std::size_t count,
    double density,
    double temperature,
    double radial_drift,
    std::uint64_t seed,
    bool spread_radii = false) {
    const ev::WeightedPICRadialMesh mesh = one_shell_mesh();
    const double marker_mass = density * mesh.shell_volume[0]
        / static_cast<double>(count);
    const double sigma = std::sqrt(temperature);
    ev::DSMCSplitMix64 rng(seed);
    std::vector<ev::WeightedPICMarker> markers(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double vx = radial_drift + sigma * gaussian(rng);
        const double vy = sigma * gaussian(rng);
        const double vz = sigma * gaussian(rng);
        ev::WeightedPICMarker& marker = markers[i];
        marker.mass = marker_mass;
        marker.r = spread_radii
            ? 0.55 + 0.9 * (static_cast<double>(i) + 0.5)
                  / static_cast<double>(count)
            : 1.0;
        marker.w = vx;
        marker.eta = 0.5 * marker.r * marker.r * (vy * vy + vz * vz);
        marker.id = static_cast<std::uint64_t>(i);
    }
    return markers;
}

ev::Params constant_parameters(double sigma) {
    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = sigma;
    p.vp_gravity = 1.0;
    return p;
}

ev::WeightedPICCollisionOptions one_shell_options(std::size_t count) {
    ev::WeightedPICCollisionOptions options;
    options.target_shell_markers = count;
    options.minimum_shell_markers = 2;
    options.collision_cfl = 100.0;
    return options;
}

std::vector<double> radial_histogram(
    const std::vector<ev::WeightedPICMarker>& markers,
    int bins,
    double center,
    double extent) {
    std::vector<double> result(static_cast<std::size_t>(bins), 0.0);
    for (const ev::WeightedPICMarker& marker : markers) {
        const int bin = static_cast<int>(
            (marker.w - center + extent) * bins / (2.0 * extent));
        if (bin >= 0 && bin < bins) {
            result[static_cast<std::size_t>(bin)] += 1.0;
        }
    }
    for (double& value : result) {
        value /= static_cast<double>(markers.size());
    }
    return result;
}

std::vector<double> tangential_speed_histogram(
    const std::vector<ev::WeightedPICMarker>& markers,
    int bins,
    double extent) {
    std::vector<double> result(static_cast<std::size_t>(bins), 0.0);
    for (const ev::WeightedPICMarker& marker : markers) {
        const double vt = std::sqrt(2.0 * marker.eta) / marker.r;
        const int bin = static_cast<int>(vt * bins / extent);
        if (bin >= 0 && bin < bins) {
            result[static_cast<std::size_t>(bin)] += 1.0;
        }
    }
    for (double& value : result) {
        value /= static_cast<double>(markers.size());
    }
    return result;
}

double reduced_maxwellian_ensemble_bias(
    std::size_t count,
    int realizations,
    const ev::Params& p,
    const ev::WeightedPICCollisionOptions& supplied_options) {
    constexpr int bins = 28;
    constexpr double radial_extent = 5.5;
    constexpr double tangential_extent = 6.0;
    constexpr double drift = 0.35;
    std::vector<double> radial_delta(bins, 0.0);
    std::vector<double> tangential_delta(bins, 0.0);
    ev::WeightedPICCollisionOptions options = supplied_options;
    options.target_shell_markers = count;
    options.minimum_shell_markers = 2;
    options.collision_cfl = 100.0;
    for (int realization = 0; realization < realizations; ++realization) {
        std::vector<ev::WeightedPICMarker> markers = maxwellian_markers(
            count,
            1.0,
            1.0,
            drift,
            UINT64_C(0x12340000) + static_cast<std::uint64_t>(realization));
        const std::vector<double> radial_before = radial_histogram(
            markers, bins, drift, radial_extent);
        const std::vector<double> tangential_before =
            tangential_speed_histogram(markers, bins, tangential_extent);
        ev::WeightedPICCollisionSamplerState sampler{
            UINT64_C(0xabc00000) + static_cast<std::uint64_t>(realization), 0};
        ev::WeightedPICCollisionWorkspace workspace;
        ev::apply_weighted_pic_collision_interval(
            markers,
            one_shell_mesh(),
            0.35,
            p,
            options,
            sampler,
            workspace);
        const std::vector<double> radial_after = radial_histogram(
            markers, bins, drift, radial_extent);
        const std::vector<double> tangential_after =
            tangential_speed_histogram(markers, bins, tangential_extent);
        for (int bin = 0; bin < bins; ++bin) {
            radial_delta[static_cast<std::size_t>(bin)] +=
                (radial_after[static_cast<std::size_t>(bin)]
                 - radial_before[static_cast<std::size_t>(bin)])
                / realizations;
            tangential_delta[static_cast<std::size_t>(bin)] +=
                (tangential_after[static_cast<std::size_t>(bin)]
                 - tangential_before[static_cast<std::size_t>(bin)])
                / realizations;
        }
    }
    double l1 = 0.0;
    for (int bin = 0; bin < bins; ++bin) {
        l1 += std::abs(radial_delta[static_cast<std::size_t>(bin)]);
        l1 += std::abs(tangential_delta[static_cast<std::size_t>(bin)]);
    }
    return l1;
}

void test_pair_rate_and_invariants() {
    constexpr std::size_t count = 180000;
    constexpr double density = 1.3;
    constexpr double temperature = 0.7;
    constexpr double dt = 0.45;
    const ev::Params p = constant_parameters(0.7);
    std::vector<ev::WeightedPICMarker> markers = maxwellian_markers(
        count, density, temperature, 0.4, UINT64_C(0x987654321));
    const ev::WeightedPICRadialMesh mesh = one_shell_mesh();
    ev::WeightedPICCollisionOptions options = one_shell_options(count);
    ev::WeightedPICCollisionSamplerState sampler{UINT64_C(0x102030405), 0};
    ev::WeightedPICCollisionWorkspace workspace;
    const ev::WeightedPICCollisionDiagnostics stats =
        ev::apply_weighted_pic_collision_interval(
            markers, mesh, dt, p, options, sampler, workspace);

    const double mean_relative_speed =
        4.0 * std::sqrt(temperature / ev::PI);
    const double exact_event_rate_density =
        0.5 * density * density * p.bp_sigma0 * mean_relative_speed;
    const double sampled_event_rate_density =
        stats.represented_event_mass / (mesh.shell_volume[0] * dt);
    const double relative_error = std::abs(
        sampled_event_rate_density - exact_event_rate_density)
        / exact_event_rate_density;
    std::cout << "PIC collision rate relative error=" << relative_error
              << " accepted=" << stats.accepted_pairs << '\n';
    assert(stats.accepted_pairs > 10000);
    assert(relative_error < 0.02);
    assert(std::abs(stats.radial_momentum_error)
           < 3.0e-12 * density * mesh.shell_volume[0]);
    assert(std::abs(stats.kinetic_energy_error)
           < 3.0e-12 * density * mesh.shell_volume[0] * temperature);
    assert(stats.max_pair_momentum_error < 2.0e-12);
    assert(stats.max_pair_kinetic_energy_error < 2.0e-12);
}

void test_sampled_collision_clock_tracks_mean_kernel() {
    constexpr std::size_t count = 50000;
    constexpr double density = 1.3;
    constexpr double temperature = 0.7;
    const ev::Params p = constant_parameters(0.7);
    const std::vector<ev::WeightedPICMarker> markers = maxwellian_markers(
        count, density, temperature, 0.4, UINT64_C(0xabcdef012345));
    ev::WeightedPICCollisionOptions options = one_shell_options(count);
    options.scheduling_pair_samples_per_shell = 4096;
    ev::WeightedPICCollisionWorkspace workspace;
    const std::vector<ev::WeightedPICCollisionShell>& shells =
        ev::build_weighted_pic_collision_shells(
            markers, one_shell_mesh(), p, options, workspace);
    const double sampled = ev::weighted_pic_sampled_collision_timescale(
        markers, p, options, workspace);
    const double repeated = ev::weighted_pic_sampled_collision_timescale(
        markers, p, options, workspace);
    const double majorant =
        ev::weighted_pic_collision_majorant_timescale(shells);
    const double mean_relative_speed =
        4.0 * std::sqrt(temperature / ev::PI);
    const double partner_density =
        density * static_cast<double>(count - 1)
        / static_cast<double>(count);
    const double exact = 1.0
        / (partner_density * p.bp_sigma0 * mean_relative_speed);
    const double relative_error = std::abs(sampled - exact) / exact;
    std::cout << "PIC sampled collision clock relative error="
              << relative_error << " majorant_ratio=" << majorant / exact
              << '\n';
    assert(sampled == repeated);
    assert(sampled >= majorant);
    assert(relative_error < 0.09);
}

void test_velocity_averaged_mean_free_path_recovers_frank_and_evolves() {
    constexpr std::size_t count = 32;
    constexpr double density = 1.7;
    const ev::WeightedPICRadialMesh mesh = one_shell_mesh();
    const double marker_mass =
        density * mesh.shell_volume[0] / static_cast<double>(count);
    std::vector<ev::WeightedPICMarker> slow(count);
    for (std::size_t i = 0; i < count; ++i) {
        slow[i].mass = marker_mass;
        slow[i].r = 1.0;
        slow[i].w = i % 2 == 0 ? -0.2 : 0.2;
        slow[i].eta = 0.5 * 0.1 * 0.1;
        slow[i].id = static_cast<std::uint64_t>(i);
    }

    const ev::Params constant = constant_parameters(2.0);
    const ev::WeightedPICCollisionOptions options = one_shell_options(count);
    ev::WeightedPICCollisionWorkspace workspace;
    const auto& constant_shells = ev::build_weighted_pic_collision_shells(
        slow, mesh, constant, options, workspace);
    assert(constant_shells.size() == 1);
    const ev::WeightedPICCollisionShell frank = constant_shells.front();
    const double frank_mean_free_path =
        static_cast<double>(count)
        / static_cast<double>(count - 1)
        / (density * constant.bp_sigma0);
    assert(std::abs(frank.mean_free_path - frank_mean_free_path)
           < 2.0e-14 * frank_mean_free_path);
    assert(frank.mean_free_path_pair_samples == count * (count - 1) / 2);
    assert(std::abs(
               frank.orbital_velocity_dispersion
                   - std::sqrt((0.2 * 0.2 + 0.1 * 0.1) / 3.0))
           < 2.0e-14);

    ev::Params velocity_dependent = constant;
    velocity_dependent.bp_cross_section = "powerlaw";
    velocity_dependent.bp_sigma_u0 = 0.05;
    velocity_dependent.bp_sigma_n = 4.0;
    const auto& slow_shells = ev::build_weighted_pic_collision_shells(
        slow, mesh, velocity_dependent, options, workspace);
    const ev::WeightedPICCollisionShell slow_shell = slow_shells.front();
    std::vector<ev::WeightedPICMarker> fast = slow;
    for (auto& marker : fast) {
        marker.w *= 4.0;
        marker.eta *= 16.0;
    }
    const auto& fast_shells = ev::build_weighted_pic_collision_shells(
        fast, mesh, velocity_dependent, options, workspace);
    const ev::WeightedPICCollisionShell fast_shell = fast_shells.front();
    assert(std::abs(
               fast_shell.orbital_velocity_dispersion
                   - 4.0 * slow_shell.orbital_velocity_dispersion)
           < 2.0e-14);
    assert(fast_shell.mean_free_path > slow_shell.mean_free_path);
}

void test_maxwellian_is_invariant_in_reduced_coordinates() {
    const ev::Params p = constant_parameters(1.0);
    const ev::WeightedPICCollisionOptions options = one_shell_options(2);
    const double small = reduced_maxwellian_ensemble_bias(
        12000, 20, p, options);
    const double large = reduced_maxwellian_ensemble_bias(
        48000, 20, p, options);
    std::cout << "PIC reduced Maxwellian histogram bias: N=12000 "
              << small << ", N=48000 " << large << '\n';
    assert(large < 0.78 * small);
    assert(large < 0.009);
}

void test_two_stream_relaxes() {
    constexpr std::size_t count = 100000;
    ev::Params p = constant_parameters(1.0);
    std::vector<ev::WeightedPICMarker> markers = maxwellian_markers(
        count, 1.0, 0.04, 0.0, UINT64_C(0xfeedface));
    for (std::size_t i = 0; i < markers.size(); ++i) {
        markers[i].w += (i & 1U) ? 2.0 : -2.0;
    }
    auto anisotropy = [](const std::vector<ev::WeightedPICMarker>& sample) {
        long double radial = 0.0L;
        long double tangential = 0.0L;
        for (const ev::WeightedPICMarker& marker : sample) {
            radial += marker.mass * marker.w * marker.w;
            tangential += marker.mass * marker.eta
                / (marker.r * marker.r);
        }
        return static_cast<double>(radial - tangential);
    };
    const double before = anisotropy(markers);
    ev::WeightedPICCollisionSamplerState sampler{UINT64_C(0xcafebabe), 0};
    ev::WeightedPICCollisionWorkspace workspace;
    ev::apply_weighted_pic_collision_interval(
        markers,
        one_shell_mesh(),
        0.35,
        p,
        one_shell_options(count),
        sampler,
        workspace);
    const double after = anisotropy(markers);
    std::cout << "PIC two-stream anisotropy ratio=" << after / before << '\n';
    assert(std::abs(after) < 0.72 * std::abs(before));
}

void test_yukawa_angular_limits_and_cdf() {
    ev::Params p = constant_parameters(1.0);
    p.bp_cross_section = "yukawa_born";
    p.bp_boltzmann_kernel = "yukawa_born";
    p.bp_yukawa_mchi = 1.0;
    p.bp_yukawa_mphi = 1.0;
    ev::WeightedPICCollisionOptions options;
    const double quantiles[] = {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99};
    // A nearly constant differential cross section has uniform mu. The
    // analytic anisotropic correction is bounded by 2*y, with y=(u/uY)^2.
    // This tests the physical limit where the former subtractive inverse
    // lost up to a percent in mu.
    for (double y : {2.02e-14, 2.0e-13, 2.0e-12, 1.0e-10, 0.0}) {
        for (double z : quantiles) {
            const double mu =
                ev::weighted_pic_collision_detail::sample_scatter_mu_from_unit(
                    std::sqrt(y), p, options, z);
            assert(std::abs(mu - (1.0 - 2.0 * z)) < 2.0 * y + 2.0e-15);
        }
    }
    // Integrating the differential law from mu to +1 gives the survival
    // probability z=(1-mu)*(1+y)/(2+y*(1-mu)). Check that independent CDF.
    for (double y : {0.01, 0.1, 1.0, 10.0, 1.0e4}) {
        double previous = 1.0;
        for (double z : quantiles) {
            const double mu =
                ev::weighted_pic_collision_detail::sample_scatter_mu_from_unit(
                    std::sqrt(y), p, options, z);
            const long double delta = 1.0L - mu;
            const long double probability =
                delta * (1.0L + y) / (2.0L + y * delta);
            assert(std::abs(probability - z) < 2.0e-11L);
            assert(mu <= previous && mu >= -1.0);
            previous = mu;
        }
        // SphDSMC and WPIC must use the same angular law and consume one
        // random deviate for a Yukawa event.
        ev::DSMCSplitMix64 rng(UINT64_C(0x123456789));
        ev::DSMCSplitMix64 reference = rng;
        const double expected =
            ev::weighted_pic_collision_detail::sample_scatter_mu_from_unit(
                std::sqrt(y), p, options, reference.uniform_open());
        assert(ev::dsmc_sample_scatter_mu(std::sqrt(y), p, rng) == expected);
        assert(rng.uniform_open() == reference.uniform_open());
    }
    for (double speed : {0.0, 1.0e-200, 1.0e200}) {
        for (double z : {0.0, 0.25, 0.75, 1.0}) {
            const double mu =
                ev::weighted_pic_collision_detail::sample_scatter_mu_from_unit(
                    speed, p, options, z);
            assert(std::isfinite(mu) && mu >= -1.0 && mu <= 1.0);
        }
    }
}

void test_velocity_dependent_sidm_kernels() {
    std::vector<ev::Params> cases(2, constant_parameters(0.0));
    cases[0].bp_cross_section = "powerlaw";
    cases[0].bp_sigma0 = 0.8;
    cases[0].bp_sigma_u0 = 0.6;
    cases[0].bp_sigma_n = 4.0;
    cases[1].bp_cross_section = "yukawa_born";
    cases[1].bp_boltzmann_kernel = "yukawa_born";
    cases[1].bp_yukawa_alpha = 0.2;
    cases[1].bp_yukawa_mchi = 1.1;
    cases[1].bp_yukawa_mphi = 0.7;

    for (std::size_t model = 0; model < cases.size(); ++model) {
        std::vector<ev::WeightedPICMarker> markers = maxwellian_markers(
            24000,
            1.0,
            0.8,
            0.2,
            UINT64_C(0x11223344) + model);
        ev::WeightedPICCollisionSamplerState sampler{
            UINT64_C(0x55667788) + model, 0};
        ev::WeightedPICCollisionWorkspace workspace;
        const ev::WeightedPICCollisionDiagnostics stats =
            ev::apply_weighted_pic_collision_interval(
                markers,
                one_shell_mesh(),
                0.25,
                cases[model],
                one_shell_options(markers.size()),
                sampler,
                workspace);
        assert(stats.accepted_pairs > 100);
        assert(std::abs(stats.radial_momentum_error) < 2.0e-12);
        assert(std::abs(stats.kinetic_energy_error) < 2.0e-12);
        assert(stats.max_pair_momentum_error < 2.0e-12);
        assert(stats.max_pair_kinetic_energy_error < 2.0e-12);
    }
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
}

void test_shell_parallelism_is_bitwise_deterministic() {
    constexpr std::size_t count = 8192;
    const ev::Params p = constant_parameters(2.0);
    std::vector<ev::WeightedPICMarker> serial = maxwellian_markers(
        count, 1.0, 0.8, 0.0, UINT64_C(0x123456), true);
    std::vector<ev::WeightedPICMarker> parallel = serial;
    ev::WeightedPICCollisionOptions options;
    options.target_shell_markers = 512;
    options.minimum_shell_markers = 16;
    options.collision_cfl = 100.0;
    ev::WeightedPICCollisionSamplerState serial_sampler{UINT64_C(77), 9};
    ev::WeightedPICCollisionSamplerState parallel_sampler = serial_sampler;
    ev::WeightedPICCollisionWorkspace serial_workspace;
    ev::WeightedPICCollisionWorkspace parallel_workspace;
    const ev::WeightedPICCollisionDiagnostics serial_stats =
        ev::apply_weighted_pic_collision_interval(
            serial, one_shell_mesh(), 0.2, p, options,
            serial_sampler, serial_workspace, 1);
    const ev::WeightedPICCollisionDiagnostics parallel_stats =
        ev::apply_weighted_pic_collision_interval(
            parallel, one_shell_mesh(), 0.2, p, options,
            parallel_sampler, parallel_workspace, 4);
    assert(serial_sampler.next_counter == parallel_sampler.next_counter);
    assert(serial_stats.trial_pairs == parallel_stats.trial_pairs);
    assert(serial_stats.accepted_pairs == parallel_stats.accepted_pairs);
    assert(serial_stats.radial_momentum_error
           == parallel_stats.radial_momentum_error);
    assert(serial_stats.kinetic_energy_error
           == parallel_stats.kinetic_energy_error);
    for (std::size_t i = 0; i < count; ++i) {
        assert_same_marker(serial[i], parallel[i]);
    }
    std::cout << "PIC collision thread determinism: shells="
              << serial_workspace.shells.size()
              << " accepted=" << serial_stats.accepted_pairs << '\n';
}

void test_parallel_radix_order_matches_serial_r_id_order() {
    constexpr std::size_t count = 32768;
    const ev::Params p = constant_parameters(2.0);
    std::vector<ev::WeightedPICMarker> markers = maxwellian_markers(
        count, 1.0, 0.8, 0.0, UINT64_C(0x99112233), true);
    for (std::size_t i = 0; i < markers.size(); ++i) {
        markers[i].r = 0.55 + 0.9
            * static_cast<double>((37 * i) % 257) / 256.0;
        markers[i].id = static_cast<std::uint64_t>(
            (8191 * i + 17) % count);
        if (i % 101 == 0) {
            markers[i].state = ev::WeightedPICMarkerState::escaped;
        }
    }
    ev::WeightedPICCollisionOptions options;
    options.target_shell_markers = 512;
    options.minimum_shell_markers = 16;
    ev::WeightedPICCollisionWorkspace serial;
    ev::WeightedPICCollisionWorkspace parallel;
    ev::build_weighted_pic_collision_shells(
        markers, one_shell_mesh(), p, options, serial,
        nullptr, nullptr, 1);
    ev::build_weighted_pic_collision_shells(
        markers, one_shell_mesh(), p, options, parallel,
        nullptr, nullptr, 8);
    assert(serial.active_by_radius == parallel.active_by_radius);
    assert(serial.shells.size() == parallel.shells.size());
    for (std::size_t i = 0; i < serial.shells.size(); ++i) {
        const auto& first = serial.shells[i];
        const auto& second = parallel.shells[i];
        assert(first.begin == second.begin);
        assert(first.end == second.end);
        assert(first.inner_edge == second.inner_edge);
        assert(first.outer_edge == second.outer_edge);
        assert(first.mean_free_path == second.mean_free_path);
        assert(first.majorant_u_sigma == second.majorant_u_sigma);
    }
}

void test_reflecting_active_population_reuse_matches_fresh_build() {
    constexpr std::size_t count = 32768;
    ev::Params p = constant_parameters(2.0);
    p.outer_boundary = "reflecting";
    std::vector<ev::WeightedPICMarker> markers = maxwellian_markers(
        count, 1.0, 0.8, 0.0, UINT64_C(0x33445566), true);
    for (std::size_t i = 0; i < markers.size(); ++i) {
        markers[i].r = 0.55 + 0.9
            * static_cast<double>((29 * i) % 509) / 508.0;
        markers[i].id = static_cast<std::uint64_t>(
            (16381 * i + 29) % count);
    }
    ev::WeightedPICCollisionOptions options;
    options.target_shell_markers = 512;
    options.minimum_shell_markers = 16;
    ev::WeightedPICCollisionWorkspace reused;
    ev::build_weighted_pic_collision_shells(
        markers, one_shell_mesh(), p, options, reused,
        nullptr, nullptr, 8);
    assert(reused.active_population_is_invariant);
    for (std::size_t i = 0; i < markers.size(); ++i) {
        markers[i].r = 0.55 + 0.9
            * static_cast<double>((47 * i + 3) % 521) / 520.0;
    }
    ev::build_weighted_pic_collision_shells(
        markers, one_shell_mesh(), p, options, reused,
        nullptr, nullptr, 8);
    ev::WeightedPICCollisionWorkspace fresh;
    ev::build_weighted_pic_collision_shells(
        markers, one_shell_mesh(), p, options, fresh,
        nullptr, nullptr, 8);
    assert(reused.active_by_radius == fresh.active_by_radius);
    assert(reused.shells.size() == fresh.shells.size());
    for (std::size_t i = 0; i < reused.shells.size(); ++i) {
        assert(reused.shells[i].begin == fresh.shells[i].begin);
        assert(reused.shells[i].end == fresh.shells[i].end);
        assert(reused.shells[i].inner_edge == fresh.shells[i].inner_edge);
        assert(reused.shells[i].outer_edge == fresh.shells[i].outer_edge);
        assert(reused.shells[i].mean_free_path
               == fresh.shells[i].mean_free_path);
        assert(reused.shells[i].majorant_u_sigma
               == fresh.shells[i].majorant_u_sigma);
    }
}

void test_unequal_masses_collide_conservatively() {
    std::vector<ev::WeightedPICMarker> markers = maxwellian_markers(
        4096, 1.0, 1.0, 0.0, UINT64_C(88));
    for (std::size_t i = 0; i < markers.size(); ++i) {
        markers[i].mass *= 0.5 + static_cast<double>(i % 7) / 4.0;
    }
    const ev::Params p = constant_parameters(4.0);
    const ev::WeightedPICCollisionOptions options =
        one_shell_options(markers.size());
    const double pair_cross_section =
        ev::weighted_pic_collision_detail::pair_total_cross_section(
            1.0, 1.0, 3.0, p, options);
    assert(std::abs(pair_cross_section - 2.0 * p.bp_sigma0) < 1.0e-14);
    ev::WeightedPICCollisionSamplerState sampler;
    ev::WeightedPICCollisionWorkspace workspace;
    const auto stats = ev::apply_weighted_pic_collision_interval(
        markers,
        one_shell_mesh(),
        0.2,
        p,
        options,
        sampler,
        workspace);
    assert(stats.accepted_pairs > 100);
    assert(stats.maximum_relative_weight_spread > 0.0);
    assert(std::abs(stats.radial_momentum_error) < 2.0e-12);
    assert(std::abs(stats.kinetic_energy_error) < 2.0e-12);
    assert(stats.max_pair_momentum_error < 2.0e-12);
    assert(stats.max_pair_kinetic_energy_error < 2.0e-12);
}

void test_reused_population_revalidates_current_markers_and_domain() {
    ev::Params p = constant_parameters(2.0);
    p.outer_boundary = "reflecting";
    auto markers = maxwellian_markers(64, 1.0, 0.8, 0.0, 99, true);
    const auto options = one_shell_options(markers.size());
    auto mesh = one_shell_mesh();
    ev::WeightedPICCollisionWorkspace reused, fresh;
    ev::build_weighted_pic_collision_shells(markers, mesh, p, options, reused);
    markers[0].state = ev::WeightedPICMarkerState::escaped;
    ev::build_weighted_pic_collision_shells(markers, mesh, p, options, reused);
    ev::build_weighted_pic_collision_shells(markers, mesh, p, options, fresh);
    assert(reused.active_by_radius == fresh.active_by_radius);
    assert(reused.active_by_radius.size() == markers.size() - 1);
    markers[0].state = ev::WeightedPICMarkerState::active;
    ev::build_weighted_pic_collision_shells(markers, mesh, p, options, reused);
    assert(reused.active_population_is_invariant);
    markers[0].r = mesh.edge.back() + 1.0;
    bool rejected = false;
    try {
        ev::build_weighted_pic_collision_shells(markers, mesh, p, options, reused);
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    markers[0].r = 1.0;
    for (auto& marker : markers) marker.id = markers.size() - 1 - marker.id;
    ev::build_weighted_pic_collision_shells(markers, mesh, p, options, reused);
    ev::WeightedPICCollisionWorkspace reordered;
    ev::build_weighted_pic_collision_shells(markers, mesh, p, options, reordered);
    assert(reused.active_by_radius == reordered.active_by_radius);
}

void test_material_origin_conserves_stored_energy() {
    ev::Params p = constant_parameters(2.0);
    p.outer_boundary = "reflecting";
    auto mesh = one_shell_mesh();
    mesh.edge.front() = 0.0;
    mesh.shell_volume[0] = (4.0 * ev::PI / 3.0) * std::pow(mesh.edge.back(), 3);
    auto markers = maxwellian_markers(128, 1.0, 0.8, 0.0, 97);
    long double mass = 0.0L, energy_before = 0.0L;
    for (auto& marker : markers) {
        marker.r = 0.0;
        marker.eta = 0.0;
        mass += marker.mass;
        energy_before += 0.5L * marker.mass * marker.w * marker.w;
    }
    auto options = one_shell_options(markers.size());
    ev::WeightedPICCollisionWorkspace workspace;
    ev::build_weighted_pic_collision_shells(markers, mesh, p, options, workspace);
    const double expected_dispersion = std::sqrt(
        static_cast<double>(2.0L * energy_before / (3.0L * mass)));
    assert(std::abs(workspace.shells[0].orbital_velocity_dispersion
        - expected_dispersion) < 2.0e-15);
    ev::WeightedPICCollisionSamplerState sampler{99, 0};
    const auto diagnostics = ev::apply_weighted_pic_collision_interval(
        markers, mesh, 1.0, p, options, sampler, workspace);
    assert(diagnostics.accepted_pairs > 0);
    long double energy_after = 0.0L;
    for (const auto& marker : markers) {
        assert(std::isfinite(marker.w) && marker.eta == 0.0);
        energy_after += 0.5L * marker.mass * marker.w * marker.w;
    }
    assert(std::abs(energy_after - energy_before) < 2.0e-14L * energy_before);
    assert(diagnostics.max_pair_kinetic_energy_error < 1.0e-14);
    assert(diagnostics.radial_momentum_error == 0.0);

    // Equal speeds at an isotropic origin have <u>=4v/3, although their
    // stored outgoing radial speeds are identical. The event clock must
    // resolve this physical relative motion.
    for (auto& marker : markers) marker.w = 1.0;
    options.scheduling_pair_samples_per_shell = 16384;
    options.scheduling_standard_error_buffer = 0.0;
    ev::build_weighted_pic_collision_shells(markers, mesh, p, options, workspace);
    const double clock = ev::weighted_pic_sampled_collision_timescale(
        markers, p, options, workspace);
    const double rate = (markers.size() - 1) / mesh.shell_volume[0]
        * markers[0].mass * p.bp_sigma0 * (4.0 / 3.0);
    assert(std::isfinite(clock) && std::abs(clock * rate - 1.0) < 0.02);

    // Signed floating-point zeros represent the same physical radius.
    // The optional parallel radix path must retain the secondary ID order.
    auto zeros = maxwellian_markers(8192, 1.0, 0.8, 0.0, 95);
    for (std::size_t i = 0; i < zeros.size(); ++i) {
        zeros[i].r = i % 2 ? -0.0 : 0.0;
        zeros[i].eta = 0.0;
        zeros[i].id = zeros.size() - 1 - i;
    }
    ev::WeightedPICCollisionWorkspace zero_workspace;
    ev::build_weighted_pic_collision_shells(
        zeros, mesh, p, options, zero_workspace, nullptr, nullptr, 4);
    for (std::size_t q = 1; q < zero_workspace.active_by_radius.size(); ++q) {
        assert(zeros[zero_workspace.active_by_radius[q - 1]].id
            < zeros[zero_workspace.active_by_radius[q]].id);
    }

    // An active nonzero angular momentum at r=0 has no finite velocity.
    markers[0].eta = 1.0;
    bool rejected = false;
    try {
        ev::build_weighted_pic_collision_shells(markers, mesh, p, options, workspace);
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    markers[0].eta = 0.0;
    markers[0].r = 1.0e-200;
    rejected = false;
    try {
        ev::build_weighted_pic_collision_shells(markers, mesh, p, options, workspace);
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}

ev::WeightedPICCollisionOptions plummer_gravity_options() {
    ev::WeightedPICCollisionOptions options;
    options.physics = ev::WeightedPICCollisionPhysics::gravitational_plummer;
    options.gravitational_scatterer_mass = 0.02;
    options.gravitational_plummer_pair_softening = 0.015;
    options.gravitational_b_min = 0.0;
    options.gravitational_b_max = 0.12;
    options.collision_cfl = 100.0;
    options.minimum_shell_markers = 2;
    return options;
}

void test_plummer_deflection_function() {
    ev::Params p = constant_parameters(0.0);
    p.bp_collision_strength = 1.0;
    ev::WeightedPICCollisionOptions options = plummer_gravity_options();

    // The exact softened quadrature tends to Rutherford scattering when the
    // pair softening is much smaller than the impact parameter.
    options.gravitational_plummer_pair_softening = 1.0e-6;
    constexpr double relative_speed = 1.3;
    for (const double impact : {0.025, 0.05, 0.1}) {
        const double measured =
            ev::weighted_pic_collision_detail::plummer_deflection_angle(
                impact, relative_speed, p, options);
        const double point_mass = 2.0 * std::atan(
            2.0 * p.vp_gravity * options.gravitational_scatterer_mass
            / (impact * relative_speed * relative_speed));
        assert(std::abs(measured - point_mass) < 2.0e-6);
    }

    // In the weak-scattering limit, the exact orbit integral approaches the
    // Plummer impulse result, and the finite core makes the b->0 deflection
    // vanish instead of producing a point-mass reversal.
    options.gravitational_scatterer_mass = 1.0e-5;
    options.gravitational_plummer_pair_softening = 0.02;
    constexpr double weak_b = 0.03;
    const double weak =
        ev::weighted_pic_collision_detail::plummer_deflection_angle(
            weak_b, relative_speed, p, options);
    const double gravitational_length = 2.0 * p.vp_gravity
        * options.gravitational_scatterer_mass
        / (relative_speed * relative_speed);
    const double impulse = 2.0 * gravitational_length * weak_b
        / (weak_b * weak_b
           + options.gravitational_plummer_pair_softening
               * options.gravitational_plummer_pair_softening);
    assert(std::abs(weak - impulse) < 0.002 * impulse);

    options = plummer_gravity_options();
    const double core =
        ev::weighted_pic_collision_detail::plummer_deflection_angle(
            1.0e-5, relative_speed, p, options);
    const double around_core =
        ev::weighted_pic_collision_detail::plummer_deflection_angle(
            options.gravitational_plummer_pair_softening,
            relative_speed,
            p,
            options);
    assert(core < around_core);
}

void test_plummer_area_sampling_and_maxwellian() {
    ev::Params p = constant_parameters(0.0);
    p.bp_collision_strength = 1.0;
    ev::WeightedPICCollisionOptions options = plummer_gravity_options();
    options.gravitational_b_min = 0.01;

    ev::DSMCSplitMix64 rng(UINT64_C(0x314159265));
    constexpr int samples = 200000;
    long double mean_b2 = 0.0L;
    for (int i = 0; i < samples; ++i) {
        const double impact =
            ev::weighted_pic_collision_detail::sample_impact_parameter(
                options, rng);
        mean_b2 += impact * impact;
    }
    mean_b2 /= samples;
    const double expected_b2 = 0.5
        * (options.gravitational_b_min * options.gravitational_b_min
           + options.gravitational_b_max * options.gravitational_b_max);
    assert(std::abs(static_cast<double>(mean_b2) - expected_b2)
           < 0.005 * expected_b2);

    const double small = reduced_maxwellian_ensemble_bias(
        6000, 12, p, options);
    const double large = reduced_maxwellian_ensemble_bias(
        24000, 12, p, options);
    std::cout << "PIC Plummer-gravity Maxwellian histogram bias: N=6000 "
              << small << ", N=24000 " << large << '\n';
    assert(large < 0.82 * small);
    assert(large < 0.017);
}

}  // namespace

int main() {
    test_yukawa_angular_limits_and_cdf();
    test_pair_rate_and_invariants();
    test_sampled_collision_clock_tracks_mean_kernel();
    test_velocity_averaged_mean_free_path_recovers_frank_and_evolves();
    test_maxwellian_is_invariant_in_reduced_coordinates();
    test_two_stream_relaxes();
    test_velocity_dependent_sidm_kernels();
    test_shell_parallelism_is_bitwise_deterministic();
    test_parallel_radix_order_matches_serial_r_id_order();
    test_reflecting_active_population_reuse_matches_fresh_build();
    test_reused_population_revalidates_current_markers_and_domain();
    test_material_origin_conserves_stored_energy();
    test_unequal_masses_collide_conservatively();
    test_plummer_deflection_function();
    test_plummer_area_sampling_and_maxwellian();
    std::cout << "weighted Vlasov PIC collision tests passed\n";
    return 0;
}
