// Regression suite for the local unequal-mass spherical-DSMC NTC kernel:
// rates, angular laws, deterministic streams, and pair invariants.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "ev/spherical_dsmc_collision.hpp"

namespace {

void test_open_uniform_upper_endpoint() {
    // This seed makes the first mixer output UINT64_MAX. Its top midpoint
    // rounds to one before the open-interval guard is applied.
    constexpr std::uint64_t seed = UINT64_C(0x31628af67b2131ab);
    ev::DSMCSplitMix64 words(seed);
    assert(words.next_u64() == UINT64_MAX);
    ev::DSMCSplitMix64 sequential(seed);
    const double expected = std::nextafter(1.0, 0.0);
    assert(sequential.uniform_open() == expected);
    assert(ev::dsmc_splitmix_uniform_open_at(seed, 0) == expected);
    assert(-std::log(expected) > 0.0);
}

void test_splitmix_random_access_matches_sequential_stream() {
    constexpr std::uint64_t seed = UINT64_C(0x123456789abcdef0);
    ev::DSMCSplitMix64 sequential(seed);
    for (std::uint64_t i = 0; i < 10000; ++i) {
        assert(
            sequential.uniform_open()
            == ev::dsmc_splitmix_uniform_open_at(seed, i));
    }
}

double gaussian(ev::DSMCSplitMix64& rng) {
    const double radius = std::sqrt(-2.0 * std::log(rng.uniform_open()));
    return radius * std::cos(2.0 * ev::PI * rng.uniform_open());
}

std::vector<ev::DSMCVelocity> maxwellian_markers(
    std::size_t count,
    double temperature,
    ev::DSMCVelocity drift,
    std::uint64_t seed) {
    ev::DSMCSplitMix64 rng(seed);
    const double sigma = std::sqrt(temperature);
    std::vector<ev::DSMCVelocity> result(count);
    for (ev::DSMCVelocity& v : result) {
        v = {
            drift.x + sigma * gaussian(rng),
            drift.y + sigma * gaussian(rng),
            drift.z + sigma * gaussian(rng),
        };
    }
    return result;
}

std::vector<double> component_histogram(
    const std::vector<ev::DSMCVelocity>& velocities,
    int bins,
    double extent) {
    std::vector<double> histogram(static_cast<std::size_t>(bins), 0.0);
    const double scale = static_cast<double>(bins) / (2.0 * extent);
    for (const ev::DSMCVelocity& v : velocities) {
        const double components[3] = {v.x, v.y, v.z};
        for (double value : components) {
            const int bin = static_cast<int>((value + extent) * scale);
            if (bin >= 0 && bin < bins) {
                histogram[static_cast<std::size_t>(bin)] += 1.0;
            }
        }
    }
    const double normalization = 1.0 / (3.0 * static_cast<double>(velocities.size()));
    for (double& value : histogram) {
        value *= normalization;
    }
    return histogram;
}

double ensemble_maxwellian_bias(
    std::size_t marker_count,
    int realizations,
    const ev::Params& p) {
    constexpr int bins = 28;
    constexpr double extent = 5.5;
    std::vector<double> mean_delta(static_cast<std::size_t>(bins), 0.0);
    for (int realization = 0; realization < realizations; ++realization) {
        std::vector<ev::DSMCVelocity> velocities = maxwellian_markers(
            marker_count,
            1.0,
            {},
            UINT64_C(0x12340000) + static_cast<std::uint64_t>(realization));
        const std::vector<double> before =
            component_histogram(velocities, bins, extent);
        ev::apply_dsmc_ntc_equal_weight(
            velocities,
            1.0,
            0.35,
            p,
            UINT64_C(0xabc00000) + static_cast<std::uint64_t>(realization));
        const std::vector<double> after =
            component_histogram(velocities, bins, extent);
        for (int bin = 0; bin < bins; ++bin) {
            mean_delta[static_cast<std::size_t>(bin)] +=
                (after[static_cast<std::size_t>(bin)]
                 - before[static_cast<std::size_t>(bin)])
                / static_cast<double>(realizations);
        }
    }
    double l1 = 0.0;
    for (double value : mean_delta) {
        l1 += std::abs(value);
    }
    return l1;
}

void test_pair_invariants_and_collision_rate() {
    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 0.7;

    constexpr std::size_t marker_count = 220000;
    constexpr double density = 1.3;
    constexpr double temperature = 0.7;
    constexpr double dt = 0.8;
    std::vector<ev::DSMCVelocity> velocities = maxwellian_markers(
        marker_count,
        temperature,
        {0.4, -0.2, 0.1},
        UINT64_C(0x987654321));
    const std::array<double, 4> before =
        ev::dsmc_marker_invariants(velocities);
    const ev::DSMCCollisionStats stats = ev::apply_dsmc_ntc_equal_weight(
        velocities,
        density,
        dt,
        p,
        UINT64_C(0x102030405));
    const std::array<double, 4> after =
        ev::dsmc_marker_invariants(velocities);

    const double momentum_scale =
        std::max({1.0, std::abs(before[0]), std::abs(before[1]), std::abs(before[2])});
    assert(std::abs(after[0] - before[0]) < 2.0e-12 * momentum_scale);
    assert(std::abs(after[1] - before[1]) < 2.0e-12 * momentum_scale);
    assert(std::abs(after[2] - before[2]) < 2.0e-12 * momentum_scale);
    assert(std::abs(after[3] - before[3]) < 2.0e-12 * before[3]);
    assert(stats.max_pair_momentum_error < 1.0e-12);
    assert(stats.max_pair_kinetic_energy_error < 1.0e-12);

    const double mean_relative_speed =
        4.0 * std::sqrt(temperature / ev::PI);
    const double analytic_event_rate =
        0.5 * density * density
        * (1.0 - 1.0 / static_cast<double>(marker_count))
        * p.bp_sigma0 * mean_relative_speed;
    const double sampled_event_rate = stats.represented_event_density / dt;
    const double relative_rate_error =
        std::abs(sampled_event_rate - analytic_event_rate) / analytic_event_rate;
    std::cout << "NTC Maxwellian collision rate relative error="
              << relative_rate_error << " accepted=" << stats.accepted_pairs << '\n';
    assert(relative_rate_error < 0.015);
}

void test_unequal_mass_pair_law_and_invariants() {
    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 0.8;
    assert(std::abs(
        ev::dsmc_pair_total_cross_section(1.0, 1.0, 3.0, p)
        - 2.0 * p.bp_sigma0) < 1.0e-14);

    constexpr std::size_t count = 40000;
    std::vector<ev::DSMCVelocity> velocities = maxwellian_markers(
        count, 0.9, {0.2, -0.1, 0.05}, UINT64_C(0x77778888));
    std::vector<double> masses(count);
    for (std::size_t i = 0; i < count; ++i) {
        masses[i] = 0.25 + 0.25 * static_cast<double>(i % 8);
    }
    const auto before = ev::dsmc_marker_invariants(velocities, masses);
    const auto stats = ev::apply_dsmc_ntc_unequal_mass(
        velocities,
        masses,
        static_cast<double>(count),
        0.5,
        p,
        UINT64_C(0x9999aaaa));
    const auto after = ev::dsmc_marker_invariants(velocities, masses);
    assert(stats.accepted_pairs > 1000);
    const double momentum_scale = std::max({
        1.0,
        std::abs(before[0]),
        std::abs(before[1]),
        std::abs(before[2]),
    });
    assert(std::abs(after[0] - before[0]) < 3.0e-12 * momentum_scale);
    assert(std::abs(after[1] - before[1]) < 3.0e-12 * momentum_scale);
    assert(std::abs(after[2] - before[2]) < 3.0e-12 * momentum_scale);
    assert(std::abs(after[3] - before[3]) < 3.0e-12 * before[3]);
    assert(stats.max_pair_momentum_error < 2.0e-12);
    assert(stats.max_pair_kinetic_energy_error < 2.0e-12);
}

void test_maxwellian_is_stationary_in_ensemble() {
    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 1.0;

    const double small_bias = ensemble_maxwellian_bias(16000, 20, p);
    const double large_bias = ensemble_maxwellian_bias(64000, 20, p);
    std::cout << "NTC Maxwellian ensemble histogram bias: N=16000 "
              << small_bias << ", N=64000 " << large_bias << '\n';
    assert(large_bias < 0.75 * small_bias);
    assert(large_bias < 0.004);
}

void test_two_stream_relaxes() {
    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 1.0;

    constexpr std::size_t marker_count = 100000;
    ev::DSMCSplitMix64 rng(UINT64_C(0xfeedface));
    std::vector<ev::DSMCVelocity> velocities(marker_count);
    for (std::size_t i = 0; i < marker_count; ++i) {
        velocities[i] = {
            ((i & 1U) ? 2.0 : -2.0) + 0.2 * gaussian(rng),
            0.2 * gaussian(rng),
            0.2 * gaussian(rng),
        };
    }
    auto anisotropy = [](const std::vector<ev::DSMCVelocity>& sample) {
        double radial = 0.0;
        double transverse = 0.0;
        for (const ev::DSMCVelocity& v : sample) {
            radial += v.x * v.x;
            transverse += 0.5 * (v.y * v.y + v.z * v.z);
        }
        return radial - transverse;
    };
    const double before = anisotropy(velocities);
    ev::apply_dsmc_ntc_equal_weight(
        velocities,
        1.0,
        0.35,
        p,
        UINT64_C(0xcafebabe));
    const double after = anisotropy(velocities);
    std::cout << "NTC two-stream anisotropy ratio=" << after / before << '\n';
    assert(std::abs(after) < 0.7 * std::abs(before));
}

void assert_majorant_bounds(
    const ev::Params& p,
    double u_max) {
    const double majorant = ev::dsmc_u_sigma_majorant(u_max, p);
    assert(majorant >= 0.0);
    constexpr int samples = 20000;
    for (int i = 0; i <= samples; ++i) {
        const double u = u_max * static_cast<double>(i)
            / static_cast<double>(samples);
        const double value = u * ev::bp_fa_total_cross_section(u, p);
        assert(value <= majorant * (1.0 + 2.0e-13) + 1.0e-300);
    }
}

void test_velocity_dependent_majorants() {
    std::vector<ev::Params> cases(4);

    cases[0].bp_cross_section = "constant";
    cases[0].bp_collision_strength = 0.8;
    cases[0].bp_sigma0 = 0.7;

    cases[1].bp_cross_section = "powerlaw";
    cases[1].bp_collision_strength = 1.3;
    cases[1].bp_sigma0 = 2.0;
    cases[1].bp_sigma_u0 = 0.3;
    cases[1].bp_sigma_n = 4.0;

    cases[2].bp_cross_section = "softened_gravity";
    cases[2].bp_collision_strength = 0.6;
    cases[2].bp_sigma0 = 0.4;
    cases[2].bp_subhalo_mass = 0.2;
    cases[2].bp_subhalo_radius = 0.3;
    cases[2].bp_coulomb_log = 2.0;
    cases[2].vp_gravity = 1.0;

    cases[3].bp_cross_section = "yukawa_born";
    cases[3].bp_collision_strength = 0.9;
    cases[3].bp_yukawa_alpha = 0.1;
    cases[3].bp_yukawa_mchi = 2.0;
    cases[3].bp_yukawa_mphi = 0.7;

    for (const ev::Params& p : cases) {
        for (double u_max : {0.01, 0.1, 1.0, 10.0, 100.0}) {
            assert_majorant_bounds(p, u_max);
        }
    }
}

void test_yukawa_differential_normalization() {
    ev::Params p;
    p.bp_cross_section = "yukawa_born";
    p.bp_boltzmann_kernel = "yukawa_born";
    p.bp_collision_strength = 0.7;
    p.bp_yukawa_alpha = 0.13;
    p.bp_yukawa_mchi = 1.8;
    p.bp_yukawa_mphi = 0.55;

    constexpr int intervals = 100000;
    for (double u : {0.03, 0.3, 1.0, 4.0, 20.0}) {
        const double a = p.bp_yukawa_mphi * p.bp_yukawa_mphi;
        const double b = 0.5 * p.bp_yukawa_mchi * p.bp_yukawa_mchi * u * u;
        const double log_extent = std::log((a + 2.0 * b) / a);
        double integral = 0.0;
        for (int i = 0; i <= intervals; ++i) {
            // Integrate uniformly in log[a + b(1-mu)].  This resolves the
            // narrow forward peak without requiring millions of uniform-mu
            // points at large relative speed.
            const double y = log_extent * static_cast<double>(i)
                / static_cast<double>(intervals);
            const double denominator = a * std::exp(y);
            const double mu = 1.0 - (denominator - a) / b;
            const double endpoint_weight =
                (i == 0 || i == intervals) ? 0.5 : 1.0;
            integral += endpoint_weight
                * ev::bp_fa_yukawa_born_dsigma_domega(mu, u, p)
                * denominator / b;
        }
        integral *= 2.0 * ev::PI * log_extent
            / static_cast<double>(intervals);
        const double total = ev::bp_fa_total_cross_section(u, p);
        const double relative = std::abs(integral - total) / total;
        assert(relative < 2.0e-9);
    }
}

}  // namespace

int main() {
    test_open_uniform_upper_endpoint();
    test_splitmix_random_access_matches_sequential_stream();
    test_pair_invariants_and_collision_rate();
    test_unequal_mass_pair_law_and_invariants();
    test_maxwellian_is_stationary_in_ensemble();
    test_two_stream_relaxes();
    test_velocity_dependent_majorants();
    test_yukawa_differential_normalization();
    std::cout << "Spherical DSMC collision tests passed.\n";
    return 0;
}
