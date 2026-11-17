#include "venus_gpu/wpic.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint64_t mix64(std::uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

double uniform_open(std::uint64_t& state) {
    state = mix64(state);
    return (static_cast<double>(state >> 11) + 0.5)
        * (1.0 / 9007199254740992.0);
}

double gaussian(std::uint64_t& state) {
    return std::sqrt(-2.0 * std::log(uniform_open(state)))
        * std::cos(2.0 * venus_gpu::pi * uniform_open(state));
}

venus_gpu::RadialMesh one_shell_mesh() {
    return venus_gpu::RadialMesh::from_edges({0.5, 1.5});
}

std::vector<venus_gpu::Marker> maxwellian_markers(
    std::size_t count,
    double density,
    double temperature,
    double drift,
    std::uint64_t seed) {
    const venus_gpu::RadialMesh mesh = one_shell_mesh();
    const double mass = density * mesh.shell_volume[0] / count;
    std::vector<venus_gpu::Marker> markers(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double vx = drift + std::sqrt(temperature) * gaussian(seed);
        const double vy = std::sqrt(temperature) * gaussian(seed);
        const double vz = std::sqrt(temperature) * gaussian(seed);
        markers[i].mass = mass;
        markers[i].r = 1.0;
        markers[i].w = vx;
        markers[i].eta = 0.5 * (vy * vy + vz * vz);
        markers[i].id = i;
    }
    return markers;
}

double radial_momentum(const std::vector<venus_gpu::Marker>& markers) {
    double result = 0.0;
    for (const venus_gpu::Marker& marker : markers) {
        result += marker.mass * marker.w;
    }
    return result;
}

double kinetic_energy(const std::vector<venus_gpu::Marker>& markers) {
    double result = 0.0;
    for (const venus_gpu::Marker& marker : markers) {
        result += 0.5 * marker.mass
            * (marker.w * marker.w + (marker.eta == 0.0 ? 0.0
                : 2.0 * marker.eta / (marker.r * marker.r)));
    }
    return result;
}

void test_material_origin_is_rejected_before_mutation() {
    // Material shells occupy the positive-radius mesh. Origin-parity
    // subhalos use their separate collision module and central volume.
    const auto mesh = one_shell_mesh();
    auto markers = maxwellian_markers(4096, 1.0, 0.8, 0.0, 97);
    for (auto& marker : markers) { marker.r = 0.0; marker.eta = 0.0; }
    venus_gpu::CollisionOptions options;
    options.sigma0 = 2.0;
    options.target_shell_markers = markers.size();
    options.minimum_shell_markers = 2;
    venus_gpu::CollisionSamplerState sampler;
    sampler.seed = 99;
    venus_gpu::WpicState state(markers);
    venus_gpu::CollisionModule collisions(mesh);
    bool rejected = false;
    try { (void)collisions.apply(state, 1.0, options, sampler); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    const auto unchanged = state.download();
    for (std::size_t i = 0; i < markers.size(); ++i) {
        assert(unchanged[i].w == markers[i].w && unchanged[i].eta == markers[i].eta);
    }
    assert(sampler.next_counter == 0);
    std::cout << "GPU material origin domain rejection preserves state\n";
}

void test_rate_invariants_and_determinism() {
    constexpr std::size_t count = 120000;
    constexpr double density = 1.3;
    constexpr double temperature = 0.7;
    constexpr double cross_section = 0.7;
    constexpr double dt = 0.35;
    const venus_gpu::RadialMesh mesh = one_shell_mesh();
    const std::vector<venus_gpu::Marker> initial = maxwellian_markers(
        count, density, temperature, 0.4, UINT64_C(0x987654321));
    const double momentum_before = radial_momentum(initial);
    const double energy_before = kinetic_energy(initial);

    venus_gpu::CollisionOptions options;
    options.sigma0 = cross_section;
    options.target_shell_markers = count;
    options.minimum_shell_markers = 2;
    options.collision_cfl = 0.15;
    venus_gpu::CollisionSamplerState sampler;
    sampler.seed = UINT64_C(0x102030405);
    venus_gpu::WpicState state(initial);
    venus_gpu::CollisionModule collisions(mesh);
    const venus_gpu::CollisionDiagnostics diagnostics =
        collisions.apply(state, dt, options, sampler);
    const std::vector<venus_gpu::Marker> after = state.download();

    const double mean_relative_speed =
        4.0 * std::sqrt(temperature / venus_gpu::pi);
    const double exact_event_mass_rate_density =
        0.5 * density * density * cross_section * mean_relative_speed;
    const double marker_mass = initial.front().mass;
    const double sampled_event_mass_rate_density =
        diagnostics.accepted_pairs * marker_mass / (mesh.shell_volume[0] * dt);
    const double rate_error = std::abs(
        sampled_event_mass_rate_density - exact_event_mass_rate_density)
        / exact_event_mass_rate_density;
    const double momentum_error = std::abs(
        radial_momentum(after) - momentum_before);
    const double energy_error = std::abs(kinetic_energy(after) - energy_before);
    assert(diagnostics.accepted_pairs > 10000);
    assert(rate_error < 0.035);
    assert(momentum_error < 2.0e-11);
    assert(energy_error < 2.0e-11);

    venus_gpu::WpicState repeated(initial);
    venus_gpu::CollisionModule repeated_collisions(mesh);
    venus_gpu::CollisionSamplerState repeated_sampler;
    repeated_sampler.seed = UINT64_C(0x102030405);
    const venus_gpu::CollisionDiagnostics repeated_diagnostics =
        repeated_collisions.apply(repeated, dt, options, repeated_sampler);
    const std::vector<venus_gpu::Marker> repeated_after = repeated.download();
    assert(repeated_diagnostics.accepted_pairs == diagnostics.accepted_pairs);
    for (std::size_t i = 0; i < count; ++i) {
        assert(repeated_after[i].w == after[i].w);
        assert(repeated_after[i].eta == after[i].eta);
    }
    std::cout << "collision_rate_relative_error=" << rate_error
              << " accepted=" << diagnostics.accepted_pairs
              << " substeps=" << diagnostics.substeps
              << " momentum_error=" << momentum_error
              << " energy_error=" << energy_error << '\n';
}

void test_identity_ordered_pair_rate(std::size_t per_shell, std::size_t shells) {
    // Alternating radial streams ordered by identity expose parity-limited
    // pair graphs. Many independent equal-volume shells measure the rate
    // without averaging away the bug by randomizing the initial velocities.
    const auto mesh = one_shell_mesh();
    const double volume = mesh.shell_volume.front() / shells;
    const double marker_mass = volume / per_shell;  // rho=sigma/m=1.
    const double dt = 0.1;
    std::vector<venus_gpu::Marker> markers(per_shell * shells);
    for (std::size_t shell = 0; shell < shells; ++shell) {
        const double q = std::pow(0.5, 3) + (std::pow(1.5, 3)-std::pow(0.5, 3))
            * (static_cast<double>(shell) + 0.5) / shells;
        const double radius = std::cbrt(q);
        for (std::size_t local = 0; local < per_shell; ++local) {
            auto& marker = markers[shell * per_shell + local];
            marker.mass = marker_mass;
            marker.r = radius;
            marker.w = local % 2 == 0 ? 1.0 : -1.0;
            marker.eta = 0.0;
            marker.id = shell * per_shell + local;
        }
    }
    const double before_momentum = radial_momentum(markers);
    const double before_energy = kinetic_energy(markers);
    venus_gpu::CollisionOptions options;
    options.sigma0 = 1.0;
    options.target_shell_markers = per_shell;
    options.minimum_shell_markers = 2;
    options.collision_cfl = 0.5;  // One matching round: no evolution-rate correction.
    venus_gpu::CollisionSamplerState sampler;
    sampler.seed = UINT64_C(0x6d61746368696e67);
    venus_gpu::WpicState state(markers);
    venus_gpu::CollisionModule collisions(mesh);
    const auto diagnostics = collisions.apply(state, dt, options, sampler);
    const auto after = state.download();
    const std::size_t plus = (per_shell + 1) / 2;
    const std::size_t minus = per_shell / 2;
    const double expected = shells * (2.0 * plus * minus / per_shell) * dt;
    const double measured = static_cast<double>(diagnostics.accepted_pairs);
    // The two domain-end shells have slightly different floating-point faces;
    // their relative contribution is below 1/shells. A seven-percent envelope
    // accommodates random factorization correlations within one shell.
    const double relative_error = std::abs(measured / expected - 1.0);
    std::cout << "identity_ordered_pair_rate N=" << per_shell
              << " shells=" << shells << " expected=" << expected
              << " accepted=" << measured << " relative_error=" << relative_error
              << " momentum_error=" << std::abs(radial_momentum(after)-before_momentum)
              << " energy_error=" << std::abs(kinetic_energy(after)-before_energy)
              << std::endl;
    assert(diagnostics.shells == shells && diagnostics.substeps == 1);
    assert(relative_error < 0.07);
    assert(std::abs(radial_momentum(after) - before_momentum) < 2.0e-10);
    assert(std::abs(kinetic_energy(after) - before_energy) < 2.0e-10);
    for (std::size_t i = 0; i < markers.size(); ++i) {
        assert(after[i].mass == markers[i].mass && after[i].r == markers[i].r);
        assert(after[i].id == markers[i].id && after[i].state == markers[i].state);
    }
}

void test_unequal_mass_conservation() {
    constexpr std::size_t count = 50000;
    const venus_gpu::RadialMesh mesh = one_shell_mesh();
    std::vector<venus_gpu::Marker> initial = maxwellian_markers(
        count, 1.0, 0.6, -0.2, UINT64_C(0x554e455155414c));
    // Preserve total represented mass while creating a deliberately broad
    // alternating pair-mass distribution.  This exercises the symmetric
    // Sigma_ij=(m_i+m_j)(sigma/m)/2 closure and the unequal-mass COM map.
    for (std::size_t i = 0; i < count; ++i) {
        initial[i].mass *= (i % 2 == 0) ? 0.25 : 1.75;
    }
    const double momentum_before = radial_momentum(initial);
    const double energy_before = kinetic_energy(initial);
    venus_gpu::CollisionOptions options;
    options.sigma0 = 1.0;
    options.target_shell_markers = count;
    options.minimum_shell_markers = 2;
    options.collision_cfl = 0.1;
    venus_gpu::CollisionSamplerState sampler;
    sampler.seed = UINT64_C(0x756e657175616c);
    venus_gpu::WpicState state(initial);
    venus_gpu::CollisionModule collisions(mesh);
    const auto diagnostics = collisions.apply(state, 0.25, options, sampler);
    const auto after = state.download();
    assert(diagnostics.accepted_pairs > 1000);
    assert(std::abs(radial_momentum(after) - momentum_before) < 2.0e-11);
    assert(std::abs(kinetic_energy(after) - energy_before) < 2.0e-11);
}

void test_yukawa_born_cross_section_and_invariants() {
    venus_gpu::CollisionOptions options;
    options.cross_section = venus_gpu::CrossSectionModel::yukawa_born;
    options.collision_strength = 0.7;
    options.yukawa_alpha = 0.031;
    options.yukawa_mchi = 1.8;
    options.yukawa_mphi = 0.42;
    const double speed = 0.73;
    const double mphi2 = options.yukawa_mphi * options.yukawa_mphi;
    const double x = options.yukawa_mchi * options.yukawa_mchi
        * speed * speed / mphi2;
    const double expected = options.collision_strength * 4.0 * venus_gpu::pi
        * options.yukawa_alpha * options.yukawa_alpha
        * options.yukawa_mchi * options.yukawa_mchi
        / (mphi2 * mphi2 * (1.0 + x));
    const double measured =
        venus_gpu::material_cross_section_per_mass(speed, options);
    assert(std::abs(measured - expected)
        < 32.0 * 2.2204460492503131e-16 * expected);

    constexpr std::size_t count = 50000;
    const venus_gpu::RadialMesh mesh = one_shell_mesh();
    const auto initial = maxwellian_markers(
        count, 1.0, 0.5, 0.1, UINT64_C(0x594b42524e));
    const double momentum_before = radial_momentum(initial);
    const double energy_before = kinetic_energy(initial);
    options.target_shell_markers = count;
    options.minimum_shell_markers = 2;
    options.collision_cfl = 0.1;
    venus_gpu::CollisionSamplerState sampler;
    sampler.seed = UINT64_C(0x594b42524e534545);
    venus_gpu::WpicState state(initial);
    venus_gpu::CollisionModule collisions(mesh);
    const auto diagnostics = collisions.apply(state, 0.1, options, sampler);
    const auto after = state.download();
    // This parameter point is deliberately forward-peaked and has a modest
    // total Born area.  A few dozen events are sufficient to exercise the
    // inverse angular CDF while the analytic cross-section assertion above is
    // the precise normalization test.
    assert(diagnostics.accepted_pairs > 10);
    assert(std::abs(radial_momentum(after) - momentum_before) < 2.0e-11);
    assert(std::abs(kinetic_energy(after) - energy_before) < 2.0e-11);
}

void test_shell_and_option_edge_validation() {
    const venus_gpu::RadialMesh mesh = one_shell_mesh();
    std::vector<venus_gpu::Marker> markers(16);
    for (std::size_t i = 0; i < markers.size(); ++i) {
        markers[i].mass = 1.0 / markers.size();
        markers[i].r = 1.0;  // Exact tie group larger than the target shell.
        markers[i].w = 0.01 * static_cast<double>(i);
        markers[i].eta = 0.01;
        markers[i].id = i;
    }
    venus_gpu::CollisionOptions options;
    options.sigma0 = 0.1;
    options.target_shell_markers = 4;
    options.minimum_shell_markers = 2;
    venus_gpu::CollisionSamplerState sampler;
    venus_gpu::WpicState state(markers);
    venus_gpu::CollisionModule collisions(mesh);
    const auto diagnostics = collisions.apply(state, 1.0e-5, options, sampler);
    assert(diagnostics.shells == 1);

    auto rejected = [](auto action) {
        try {
            action();
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    assert(rejected([&] {
        venus_gpu::CollisionOptions invalid = options;
        invalid.cross_section = static_cast<venus_gpu::CrossSectionModel>(99);
        (void)venus_gpu::material_cross_section_per_mass(1.0, invalid);
    }));
    assert(rejected([&] {
        (void)venus_gpu::material_cross_section_per_mass(-1.0, options);
    }));
    // Parameters documented as irrelevant to Yukawa--Born must not reject an
    // otherwise valid model.
    venus_gpu::CollisionOptions yukawa;
    yukawa.cross_section = venus_gpu::CrossSectionModel::yukawa_born;
    yukawa.sigma0 = std::numeric_limits<double>::quiet_NaN();
    assert(std::isfinite(
        venus_gpu::material_cross_section_per_mass(1.0, yukawa)));

    markers[0].r = 2.0;
    venus_gpu::WpicState outside(markers);
    venus_gpu::CollisionModule outside_collisions(mesh);
    venus_gpu::CollisionSamplerState outside_sampler;
    assert(rejected([&] {
        (void)outside_collisions.apply(
            outside, 1.0e-5, options, outside_sampler);
    }));
}

void test_adaptive_shell_timestep_against_reference() {
    const venus_gpu::RadialMesh mesh = venus_gpu::RadialMesh::from_edges(
        {0.5, 0.8, 1.2, 1.8, 2.6});
    constexpr std::size_t per_shell = 2048;
    std::vector<venus_gpu::Marker> markers(4 * per_shell);
    for (std::size_t shell = 0; shell < 4; ++shell) {
        const double left = mesh.edge[shell];
        const double right = mesh.edge[shell + 1];
        for (std::size_t local = 0; local < per_shell; ++local) {
            auto& marker = markers[shell * per_shell + local];
            marker.mass = 1.0e-4;
            marker.r = left + (right - left)
                * (static_cast<double>(local) + 0.5) / per_shell;
            marker.w = (static_cast<int>(local % 13) - 6)
                * (0.002 + 0.001 * shell);
            const double tangential = 0.01 + 0.001 * (local % 9);
            marker.eta = 0.5 * marker.r * marker.r
                * tangential * tangential;
            marker.id = static_cast<std::uint64_t>(
                shell * per_shell + local);
        }
    }
    const double momentum_before = radial_momentum(markers);
    const double energy_before = kinetic_energy(markers);

    venus_gpu::CollisionOptions reference_options;
    reference_options.sigma0 = 1000.0;
    reference_options.target_shell_markers = per_shell;
    reference_options.minimum_shell_markers = 32;
    reference_options.collision_cfl = 0.2;
    reference_options.maximum_substeps = 512;
    venus_gpu::CollisionOptions adaptive_options = reference_options;
    adaptive_options.adaptive_shell_timestep = true;
    venus_gpu::CollisionSamplerState reference_sampler;
    reference_sampler.seed = UINT64_C(0x6d6174657269616c);
    venus_gpu::CollisionSamplerState adaptive_sampler = reference_sampler;
    venus_gpu::WpicState reference_state(markers);
    venus_gpu::WpicState adaptive_state(markers);
    venus_gpu::CollisionModule reference_collision(mesh);
    venus_gpu::CollisionModule adaptive_collision(mesh);

    const auto reference = reference_collision.apply(
        reference_state, 0.2, reference_options, reference_sampler);
    const auto adaptive = adaptive_collision.apply(
        adaptive_state, 0.2, adaptive_options, adaptive_sampler);
    const auto reference_after = reference_state.download();
    const auto adaptive_after = adaptive_state.download();
    assert(reference.accepted_pairs > 0);
    assert(adaptive.accepted_pairs > 0);
    assert(adaptive.shell_visits < reference.shell_visits);
    const double accepted_scale = std::max(
        1.0,
        0.5 * static_cast<double>(
            reference.accepted_pairs + adaptive.accepted_pairs));
    assert(std::abs(
        static_cast<double>(reference.accepted_pairs)
        - static_cast<double>(adaptive.accepted_pairs))
        / accepted_scale < 0.25);
    for (const auto* population : {&reference_after, &adaptive_after}) {
        assert(std::abs(radial_momentum(*population) - momentum_before)
            < 3.0e-11);
        assert(std::abs(kinetic_energy(*population) - energy_before)
            < 3.0e-11);
    }
    std::cout << "material_adaptive_reference_accept="
              << reference.accepted_pairs
              << " adaptive_accept=" << adaptive.accepted_pairs
              << " reference_visits=" << reference.shell_visits
              << " adaptive_visits=" << adaptive.shell_visits << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc == 2 ? argv[1] : "";
    if (argc > 2
        || (!mode.empty() && mode != "--edge-validation-only"
            && mode != "--adaptive-only" && mode != "--matching-only")) {
        std::cerr << "usage: " << argv[0]
                  << " [--edge-validation-only|--adaptive-only|--matching-only]\n";
        return 2;
    }
    venus_gpu::require_cuda_device();
    test_material_origin_is_rejected_before_mutation();
    if (mode == "--matching-only") {
        test_identity_ordered_pair_rate(32, 8192);
        test_identity_ordered_pair_rate(3, 32768);
        std::cout << "WPIC collision GPU even/odd matching tests passed.\n";
        return 0;
    }
    if (mode == "--edge-validation-only") {
        test_shell_and_option_edge_validation();
        std::cout << "WPIC collision GPU edge validation passed.\n";
        return 0;
    }
    if (mode == "--adaptive-only") {
        test_adaptive_shell_timestep_against_reference();
        std::cout << "WPIC collision GPU adaptive test passed.\n";
        return 0;
    }
    test_rate_invariants_and_determinism();
    test_identity_ordered_pair_rate(32, 8192);
    test_identity_ordered_pair_rate(3, 32768);
    test_unequal_mass_conservation();
    test_yukawa_born_cross_section_and_invariants();
    test_shell_and_option_edge_validation();
    test_adaptive_shell_timestep_against_reference();
    std::cout << "WPIC collision GPU tests passed.\n";
    return 0;
}
