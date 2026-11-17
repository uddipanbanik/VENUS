// Frank-restart bridge tests: v4/v5 unequal-mass preservation, the deliberate
// unequal-mass thinning rejection, and mass-conserving radial smoke sampling.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "ev/weighted_pic_frank_restart_import.hpp"

namespace {

std::vector<ev::SphericalDSMCParticle> source_particles(bool unequal) {
    const std::vector<double> radius = {4.0, 1.0, 6.0, 2.0, 5.0, 3.0};
    std::vector<ev::SphericalDSMCParticle> particles(radius.size());
    for (std::size_t i = 0; i < particles.size(); ++i) {
        particles[i].r = radius[i];
        particles[i].vr = -0.3 + 0.1 * static_cast<double>(i);
        particles[i].L = 0.2 + 0.05 * static_cast<double>(i);
        particles[i].id = static_cast<std::uint64_t>(i);
        particles[i].mass = unequal
            ? 0.05 * static_cast<double>(i + 1)
            : 0.25;
    }
    return particles;
}

void assert_marker_matches_source(
    const ev::WeightedPICMarker& marker,
    const ev::SphericalDSMCParticle& source) {
    assert(marker.mass == source.mass);
    assert(marker.r == source.r);
    assert(marker.w == source.vr);
    assert(marker.eta == 0.5 * source.L * source.L);
    assert(marker.id == source.id);
}

void test_v4_and_v5_full_import_preserve_individual_masses() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path()
        / "venus_weighted_pic_frank_import_test";
    std::filesystem::remove_all(directory);
    const auto particles = source_particles(true);
    const std::filesystem::path v4_path = directory / "unequal_v4.bin";
    const std::filesystem::path v5_path = directory / "unequal_v5.bin";
    ev::spherical_dsmc_write_restart(
        v4_path, particles, 0.125, 1.25, 17, -0.5, 8, 2, 0);
    std::vector<double> remainder(particles.size(), 0.125);
    ev::spherical_dsmc_write_restart(
        v5_path,
        particles,
        0.125,
        1.25,
        17,
        -0.5,
        8,
        2,
        0,
        &remainder);

    for (const std::filesystem::path& path : {v4_path, v5_path}) {
        const ev::WeightedPICFrankRestartImport imported =
            ev::import_weighted_pic_frank_restart(path);
        assert(imported.time == 1.25);
        assert(imported.step == 17);
        assert(imported.markers.size() == particles.size());
        for (const ev::WeightedPICMarker& marker : imported.markers) {
            assert(marker.id < particles.size());
            // The SphDSMC reader restores radius order.  IDs identify the
            // original records whose physical values must remain unchanged.
            assert_marker_matches_source(
                marker, particles[static_cast<std::size_t>(marker.id)]);
        }
    }

    bool rejected_thinning = false;
    try {
        (void)ev::import_weighted_pic_frank_restart(v4_path, 3);
    } catch (const std::runtime_error&) {
        rejected_thinning = true;
    }
    assert(rejected_thinning);
    std::filesystem::remove_all(directory);
}

void test_equal_mass_thinning_is_radial_and_mass_conserving() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path()
        / "venus_weighted_pic_frank_thinning_test";
    std::filesystem::remove_all(directory);
    const auto particles = source_particles(false);
    const std::filesystem::path path = directory / "equal_v4.bin";
    ev::spherical_dsmc_write_restart(
        path, particles, 0.25, 2.5, 23, -0.5, 0, 0, 0);

    const ev::WeightedPICFrankRestartImport imported =
        ev::import_weighted_pic_frank_restart(path, 3);
    assert(imported.time == 2.5);
    assert(imported.step == 23);
    assert(imported.markers.size() == 3);
    // The centers of three equal radial-rank strata are r=2,4,6.
    assert(imported.markers[0].r == 2.0);
    assert(imported.markers[1].r == 4.0);
    assert(imported.markers[2].r == 6.0);
    for (const ev::WeightedPICMarker& marker : imported.markers) {
        assert(marker.mass == 0.5);
    }
    const double imported_mass = std::accumulate(
        imported.markers.begin(),
        imported.markers.end(),
        0.0,
        [](double sum, const ev::WeightedPICMarker& marker) {
            return sum + marker.mass;
        });
    assert(imported_mass == 1.5);
    std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
    test_v4_and_v5_full_import_preserve_individual_masses();
    test_equal_mass_thinning_is_radial_and_mass_conserving();
    std::cout << "weighted PIC Frank restart-import tests passed\n";
    return 0;
}
