#include "venus_gpu/wpic.hpp"

#include "ev/weighted_vlasov_pic_subhalo_collisions.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double momentum(const std::vector<venus_gpu::Marker>& markers) {
    long double result = 0.0L;
    for (const auto& marker : markers) {
        result += static_cast<long double>(marker.mass) * marker.w;
    }
    return static_cast<double>(result);
}

double total_energy(const std::vector<venus_gpu::Marker>& markers) {
    long double result = 0.0L;
    for (const auto& marker : markers) {
        const double speed2 = marker.w * marker.w
            + (marker.r == 0.0 ? 0.0
               : 2.0 * marker.eta / (marker.r * marker.r));
        result += 0.5L * marker.mass * speed2 + marker.internal_energy;
    }
    return static_cast<double>(result);
}

double mass(const std::vector<venus_gpu::Marker>& markers) {
    long double result = 0.0L;
    for (const auto& marker : markers) {
        result += marker.mass;
    }
    return static_cast<double>(result);
}

void test_authoritative_table_parity() {
    venus_gpu::SubhaloCollisionOptions gpu;
    gpu.strength = 0.73;
    gpu.plummer_area_per_mass = 1.17;
    gpu.low_speed_regularization = 0.05;
    ev::WeightedPICSubhaloCollisionOptions cpu;
    cpu.strength = gpu.strength;
    cpu.plummer_area_per_mass = gpu.plummer_area_per_mass;
    cpu.low_speed_regularization = gpu.low_speed_regularization;

    const double masses[][2] = {
        {1.0, 1.0}, {0.5, 2.0}, {1.0e-4, 3.0}, {0.013, 0.71}};
    const double speeds[] = {0.001, 0.005, 0.02, 0.08, 0.3, 1.0, 5.0};
    for (const auto& pair : masses) {
        const double sigma1 = 0.07 + 0.03 * std::sqrt(pair[0]);
        const double sigma2 = 0.09 + 0.02 * std::sqrt(pair[1]);
        const double root1 = std::sqrt(pair[0]);
        const double root2 = std::sqrt(pair[1]);
        const double pair_sigma = std::sqrt(
            (root1 * sigma1 * sigma1 + root2 * sigma2 * sigma2)
            / (root1 + root2));
        for (double relative_speed : speeds) {
            const auto measured = venus_gpu::subhalo_cross_sections(
                relative_speed,
                pair[0],
                pair[1],
                sigma1,
                sigma2,
                gpu);
            const double expected_scatter =
                ev::weighted_pic_subhalo_scattering_cross_section(
                    relative_speed,
                    pair[0],
                    pair[1],
                    pair_sigma,
                    cpu);
            const double expected_capture =
                ev::weighted_pic_subhalo_capture_cross_section(
                    relative_speed,
                    pair[0],
                    pair[1],
                    pair_sigma,
                    cpu);
            const double scatter_scale = std::max(1.0, expected_scatter);
            const double capture_scale = std::max(1.0, expected_capture);
            assert(std::abs(measured.scattering - expected_scatter)
                < 128.0 * 2.2204460492503131e-16 * scatter_scale);
            assert(std::abs(measured.capture - expected_capture)
                < 128.0 * 2.2204460492503131e-16 * capture_scale);
            assert(measured.pair_velocity_scale == pair_sigma);
        }
    }
}

void test_unequal_mass_scattering_capture_ledgers() {
    const venus_gpu::RadialMesh mesh =
        venus_gpu::RadialMesh::from_edges({0.5, 1.5});
    constexpr std::size_t count = 20000;
    std::vector<venus_gpu::Marker> markers(count);
    const double base_mass = mesh.shell_volume[0] / count;
    for (std::size_t i = 0; i < count; ++i) {
        auto& marker = markers[i];
        marker.mass = base_mass * (i % 2 == 0 ? 0.35 : 1.65);
        marker.r = 0.75 + 0.5 * (static_cast<double>(i) + 0.5) / count;
        marker.w = (static_cast<int>(i % 7) - 3) * 0.006;
        const double tangential = 0.012 + 0.001 * (i % 5);
        marker.eta = 0.5 * marker.r * marker.r
            * tangential * tangential;
        marker.id = i;
        marker.source_cell = 0;
        marker.internal_velocity_dispersion = 0.1;
        marker.internal_energy = 1.5 * marker.mass * 0.1 * 0.1;
    }
    const double mass_before = mass(markers);
    const double momentum_before = momentum(markers);
    const double energy_before = total_energy(markers);

    venus_gpu::SubhaloCollisionOptions options;
    options.strength = 0.1;
    options.plummer_area_per_mass = 1.0;
    options.collision_cfl = 0.15;
    options.maximum_substeps = 512;
    options.collision_lanes = 4;
    venus_gpu::CollisionSamplerState sampler;
    sampler.seed = UINT64_C(0x73756268616c6f31);
    venus_gpu::WpicState state(markers);
    venus_gpu::SubhaloCollisionModule collisions(mesh);
    const auto diagnostics = collisions.apply(state, 0.02, options, sampler);
    const auto after = state.download();

    assert(diagnostics.scattering_events > 0);
    assert(diagnostics.capture_events > 0);
    assert(diagnostics.removed_markers == diagnostics.capture_events);
    assert(after.size() + diagnostics.capture_events == markers.size());
    assert(std::abs(mass(after) - mass_before) < 2.0e-12);
    assert(std::abs(momentum(after) - momentum_before) < 2.0e-12);
    assert(std::abs(total_energy(after) - energy_before) < 2.0e-12);
    assert(diagnostics.maximum_pair_momentum_error < 2.0e-12);
    assert(diagnostics.maximum_pair_total_energy_error < 2.0e-12);
    assert(std::abs(diagnostics.minimum_pair_velocity_scale - 0.1)
        < 2.0e-15);
    assert(std::abs(diagnostics.maximum_pair_velocity_scale - 0.1)
        < 2.0e-15);
    std::cout << "subhalo_scatter=" << diagnostics.scattering_events
              << " capture=" << diagnostics.capture_events
              << " substeps=" << diagnostics.substeps
              << " markers_after=" << after.size()
              << " pair_velocity_scale_min="
              << diagnostics.minimum_pair_velocity_scale
              << " pair_velocity_scale_max="
              << diagnostics.maximum_pair_velocity_scale << '\n';
}

void test_origin_collision_preserves_stored_energy() {
    const auto mesh = venus_gpu::RadialMesh::from_edges({0.99, 1.01});
    for (int origin_count : {1, 2}) {
        for (bool capture : {false, true}) {
            std::vector<venus_gpu::Marker> markers(2);
            for (std::size_t i = 0; i < markers.size(); ++i) {
                auto& value = markers[i];
                value.mass = i == 0 ? 0.1 : 0.17;
                value.r = i == 0 || origin_count == 2 ? 0.0 : 0.2;
                value.w = capture ? (i == 0 ? 0.01 : 0.02)
                                  : (i == 0 ? 0.08 : 0.11);
                value.eta = 0.0;
                value.id = i + 1;
                value.internal_velocity_dispersion = 0.1;
                value.internal_energy = 1.5 * value.mass * 0.01;
            }
            const double before = total_energy(markers);
            venus_gpu::SubhaloCollisionOptions options;
            options.enable_capture = capture;
            options.enable_scattering = !capture;
            options.strength = 100.0;
            options.collision_cfl = 0.2;
            options.maximum_substeps = 100000;
            options.adaptive_cell_timestep = true;
            venus_gpu::CollisionSamplerState sampler;
            sampler.seed = UINT64_C(0x54321);
            venus_gpu::WpicState state(markers);
            venus_gpu::SubhaloCollisionModule collisions(mesh);
            std::uint64_t events = 0;
            for (int interval = 0; interval < 16 && events == 0; ++interval) {
                const auto stats = collisions.apply(state, 0.5, options, sampler);
                events += stats.accepted_pairs;
                const auto after = state.download();
                assert(std::abs(total_energy(after) - before) < 1.0e-13);
                assert(std::isfinite(stats.minimum_collision_timescale));
                assert(stats.maximum_pair_momentum_error < 1.0e-13);
                assert(stats.maximum_pair_total_energy_error < 1.0e-13);
                for (const auto& value : after) {
                    if (value.r == 0.0) {
                        assert(value.w >= 0.0 && value.eta == 0.0);
                    }
                }
            }
            assert(events > 0);
            assert(state.download().size() == (capture ? 1u : 2u));
        }
    }
    std::cout << "origin scatter/capture stored-energy tests passed\n";
}

void test_invalid_internal_state_is_rejected() {
    const venus_gpu::RadialMesh mesh =
        venus_gpu::RadialMesh::from_edges({0.5, 1.5});
    venus_gpu::SubhaloCollisionOptions options;
    options.strength = 1.0;

    auto rejected = [&](double dispersion, double internal_energy) {
        std::vector<venus_gpu::Marker> markers(2);
        for (std::size_t i = 0; i < markers.size(); ++i) {
            markers[i].mass = 0.5;
            markers[i].r = 0.8 + 0.2 * i;
            markers[i].w = i == 0 ? -0.1 : 0.1;
            markers[i].eta = 0.01;
            markers[i].id = i;
            markers[i].internal_velocity_dispersion = dispersion;
            markers[i].internal_energy = internal_energy;
        }
        venus_gpu::WpicState state(markers);
        venus_gpu::SubhaloCollisionModule collisions(mesh);
        venus_gpu::CollisionSamplerState sampler;
        try {
            (void)collisions.apply(state, 0.01, options, sampler);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };

    assert(rejected(0.0, 0.0));
    assert(rejected(0.1, 0.0));
}

void test_zero_strength_is_an_exact_no_op() {
    const venus_gpu::RadialMesh mesh =
        venus_gpu::RadialMesh::from_edges({0.5, 1.5});
    std::vector<venus_gpu::Marker> markers(2);
    for (std::size_t i = 0; i < markers.size(); ++i) {
        markers[i].mass = 0.5;
        markers[i].r = 0.8 + 0.2 * i;
        markers[i].w = i == 0 ? -0.1 : 0.1;
        markers[i].eta = 0.01;
        markers[i].id = i;
        // Collisionless restarts are allowed to omit internal subhalo state.
        markers[i].internal_velocity_dispersion = 0.0;
        markers[i].internal_energy = 0.0;
    }
    venus_gpu::WpicState state(markers);
    venus_gpu::SubhaloCollisionModule collisions(mesh);
    venus_gpu::SubhaloCollisionOptions options;
    options.strength = 0.0;
    venus_gpu::CollisionSamplerState sampler;
    const auto initial_sampler = sampler;
    const auto diagnostics = collisions.apply(state, 1.0, options, sampler);
    const auto after = state.download();

    assert(sampler.seed == initial_sampler.seed);
    assert(sampler.next_counter == initial_sampler.next_counter);
    assert(diagnostics.substeps == 0);
    assert(diagnostics.trial_pairs == 0);
    assert(diagnostics.accepted_pairs == 0);
    assert(after.size() == markers.size());
    for (std::size_t i = 0; i < markers.size(); ++i) {
        assert(after[i].mass == markers[i].mass);
        assert(after[i].r == markers[i].r);
        assert(after[i].w == markers[i].w);
        assert(after[i].eta == markers[i].eta);
        assert(after[i].id == markers[i].id);
        assert(after[i].internal_energy == markers[i].internal_energy);
        assert(after[i].internal_velocity_dispersion
            == markers[i].internal_velocity_dispersion);
    }
}

void test_sampler_counter_exhaustion_is_rejected() {
    const venus_gpu::RadialMesh mesh =
        venus_gpu::RadialMesh::from_edges({0.5, 1.5});
    std::vector<venus_gpu::Marker> markers(2);
    for (std::size_t i = 0; i < markers.size(); ++i) {
        markers[i].mass = 0.5;
        markers[i].r = 0.8 + 0.2 * i;
        markers[i].w = i == 0 ? -0.1 : 0.1;
        markers[i].eta = 0.01;
        markers[i].id = i;
        markers[i].internal_velocity_dispersion = 0.1;
        markers[i].internal_energy = 1.5 * markers[i].mass * 0.1 * 0.1;
    }
    venus_gpu::WpicState state(markers);
    venus_gpu::SubhaloCollisionModule collisions(mesh);
    venus_gpu::SubhaloCollisionOptions options;
    options.strength = 1.0;
    venus_gpu::CollisionSamplerState sampler;
    sampler.next_counter = std::numeric_limits<std::uint64_t>::max();
    bool threw = false;
    try {
        (void)collisions.apply(state, 0.01, options, sampler);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    assert(sampler.next_counter == std::numeric_limits<std::uint64_t>::max());
}

void test_adaptive_cell_timestep_against_reference() {
    const venus_gpu::RadialMesh mesh =
        venus_gpu::RadialMesh::from_edges({0.5, 0.8, 1.2, 1.8, 2.6});
    constexpr std::size_t per_cell = 4096;
    std::vector<venus_gpu::Marker> markers(4 * per_cell);
    std::uint64_t id = 0;
    for (std::size_t cell = 0; cell < 4; ++cell) {
        const double left = mesh.edge[cell];
        const double right = mesh.edge[cell + 1];
        const double base_mass = mesh.shell_volume[cell] / per_cell;
        for (std::size_t local = 0; local < per_cell; ++local) {
            auto& marker = markers[cell * per_cell + local];
            marker.mass = base_mass
                * (local % 5 == 0 ? 2.4 : (local % 2 == 0 ? 0.45 : 0.95));
            marker.r = left + (right - left)
                * (static_cast<double>(local) + 0.5) / per_cell;
            marker.w = (static_cast<int>(local % 11) - 5)
                * (0.0015 + 0.0005 * cell);
            const double tangential = 0.008 + 0.001 * (local % 7);
            marker.eta = 0.5 * marker.r * marker.r
                * tangential * tangential;
            marker.id = id++;
            marker.source_cell = cell;
            marker.internal_velocity_dispersion =
                cell == 0 ? 0.24 : (0.035 + 0.01 * cell);
            marker.internal_energy = 1.5 * marker.mass
                * marker.internal_velocity_dispersion
                * marker.internal_velocity_dispersion;
        }
    }

    venus_gpu::SubhaloCollisionOptions reference_options;
    reference_options.strength = 0.04;
    reference_options.plummer_area_per_mass = 1.0;
    reference_options.collision_cfl = 0.2;
    reference_options.maximum_substeps = 32768;
    reference_options.collision_lanes = 4;
    venus_gpu::SubhaloCollisionOptions adaptive_options = reference_options;
    adaptive_options.adaptive_cell_timestep = true;
    adaptive_options.adaptive_batch_substeps = 256;
    venus_gpu::CollisionSamplerState reference_sampler;
    reference_sampler.seed = UINT64_C(0x6164617074697665);
    venus_gpu::CollisionSamplerState adaptive_sampler = reference_sampler;
    venus_gpu::WpicState reference_state(markers);
    venus_gpu::WpicState adaptive_state(markers);
    venus_gpu::SubhaloCollisionModule reference_collision(mesh);
    venus_gpu::SubhaloCollisionModule adaptive_collision(mesh);

    const auto reference_start = std::chrono::steady_clock::now();
    const auto reference = reference_collision.apply(
        reference_state, 0.02, reference_options, reference_sampler);
    const double reference_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reference_start).count();
    const auto adaptive_start = std::chrono::steady_clock::now();
    const auto adaptive = adaptive_collision.apply(
        adaptive_state, 0.02, adaptive_options, adaptive_sampler);
    const double adaptive_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - adaptive_start).count();

    const auto reference_after = reference_state.download();
    const auto adaptive_after = adaptive_state.download();
    const double initial_mass = mass(markers);
    const double initial_momentum = momentum(markers);
    const double initial_energy = total_energy(markers);
    for (const auto* population : {&reference_after, &adaptive_after}) {
        assert(std::abs(mass(*population) - initial_mass) < 3.0e-11);
        assert(std::abs(momentum(*population) - initial_momentum) < 3.0e-11);
        assert(std::abs(total_energy(*population) - initial_energy) < 3.0e-11);
    }
    assert(reference.accepted_pairs > 0);
    assert(adaptive.accepted_pairs > 0);
    assert(reference.capture_events > 0);
    assert(adaptive.capture_events > 0);
    const double accepted_scale = std::max(
        1.0,
        0.5 * static_cast<double>(
            reference.accepted_pairs + adaptive.accepted_pairs));
    const double capture_scale = std::max(
        1.0,
        0.5 * static_cast<double>(
            reference.capture_events + adaptive.capture_events));
    assert(std::abs(
        static_cast<double>(reference.accepted_pairs)
        - static_cast<double>(adaptive.accepted_pairs))
        / accepted_scale < 0.35);
    assert(std::abs(
        static_cast<double>(reference.capture_events)
        - static_cast<double>(adaptive.capture_events))
        / capture_scale < 0.45);
    std::cout << "adaptive_mesh_reference_seconds=" << reference_seconds
              << " adaptive_seconds=" << adaptive_seconds
              << " reference_batches=" << reference.kernel_batches
              << " adaptive_batches=" << adaptive.kernel_batches
              << " reference_accept=" << reference.accepted_pairs
              << " adaptive_accept=" << adaptive.accepted_pairs
              << " reference_capture=" << reference.capture_events
              << " adaptive_capture=" << adaptive.capture_events << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc == 2 ? argv[1] : "";
    if (argc > 2
        || (!mode.empty() && mode != "--table-only"
            && mode != "--zero-strength-only"
            && mode != "--invalid-state-only"
            && mode != "--counter-exhaustion-only"
            && mode != "--scatter-capture-only"
            && mode != "--adaptive-only"
            && mode != "--origin-only")) {
        std::cerr << "usage: " << argv[0]
                  << " [--table-only|--zero-strength-only|"
                     "--invalid-state-only|--counter-exhaustion-only|"
                     "--scatter-capture-only|--adaptive-only|--origin-only]\n";
        return 2;
    }
    venus_gpu::require_cuda_device();
    if (mode.empty() || mode == "--table-only") {
        std::cerr << "[subhalo-test] table parity\n";
        test_authoritative_table_parity();
    }
    if (mode.empty() || mode == "--zero-strength-only") {
        std::cerr << "[subhalo-test] exact zero-strength no-op\n";
        test_zero_strength_is_an_exact_no_op();
    }
    if (mode.empty() || mode == "--invalid-state-only") {
        std::cerr << "[subhalo-test] invalid internal-state rejection\n";
        test_invalid_internal_state_is_rejected();
    }
    if (mode.empty() || mode == "--counter-exhaustion-only") {
        std::cerr << "[subhalo-test] sampler counter exhaustion\n";
        test_sampler_counter_exhaustion_is_rejected();
    }
    if (mode.empty() || mode == "--scatter-capture-only") {
        std::cerr << "[subhalo-test] unequal-mass scatter/capture\n";
        test_unequal_mass_scattering_capture_ledgers();
    }
    if (mode.empty() || mode == "--origin-only") {
        std::cerr << "[subhalo-test] exact-origin stored energy\n";
        test_origin_collision_preserves_stored_energy();
    }
    if (mode.empty() || mode == "--adaptive-only") {
        std::cerr << "[subhalo-test] adaptive cell timestep\n";
        test_adaptive_cell_timestep_against_reference();
    }
    std::cout << "WPIC unequal-mass subhalo GPU tests passed.\n";
    return 0;
}
