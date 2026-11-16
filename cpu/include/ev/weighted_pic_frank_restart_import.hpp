#pragma once

// Convert a spherical DSMC restart into weighted Newtonian PIC markers.
// Reading every particle preserves its radius, radial velocity, represented
// mass, identity and saved epoch. The angular variable changes from L to
// eta=L^2/2. An optional reduced diagnostic sample is restricted to equal
// weights, selected at deterministic radial ranks and rescaled to the original
// total represented mass. It is a different finite particle realization.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "ev/config.hpp"
#include "ev/spherical_dsmc_solver.hpp"
#include "ev/weighted_vlasov_pic.hpp"

namespace ev {

struct WeightedPICFrankRestartImport {
    std::vector<WeightedPICMarker> markers;
    double time = 0.0;
    std::int64_t step = 0;
};

inline WeightedPICFrankRestartImport import_weighted_pic_frank_restart(
    const std::filesystem::path& path,
    std::size_t marker_limit = 0) {
    std::vector<SphericalDSMCParticle> particles;
    const SphericalDSMCRestartHeader header =
        spherical_dsmc_read_restart(path, particles);
    // A zero marker limit imports the complete distribution. A positive
    // limit explicitly requests the reduced diagnostic population below.
    const std::size_t selected = marker_limit == 0
        ? particles.size()
        : std::min(marker_limit, particles.size());
    if (selected < 2) {
        throw std::runtime_error(
            "Frank weighted PIC import needs at least two markers.");
    }

    // Importing every marker preserves an arbitrary positive mass spectrum.
    // A reduced sample is safe only when marker number is proportional to
    // represented mass, which is the equal-mass case.
    if (selected != particles.size()) {
        const double reference_mass = particles.front().mass;
        for (const SphericalDSMCParticle& particle : particles) {
            const double scale = std::max(reference_mass, particle.mass);
            if (std::abs(particle.mass - reference_mass)
                > 64.0 * std::numeric_limits<double>::epsilon() * scale) {
                throw std::runtime_error(
                    "--marker-limit is an equal-mass diagnostic option. "
                    "Import every marker from an unequal-mass Frank/SphDSMC "
                    "restart so its phase-space mass distribution is not "
                    "biased.");
            }
        }
    }

    // Accumulate the actual source weights rather than inferring total
    // mass from the number of saved records.
    long double source_total_mass = 0.0L;
    for (const SphericalDSMCParticle& particle : particles) {
        source_total_mass += static_cast<long double>(particle.mass);
    }
    if (!(source_total_mass > 0.0L)) {
        throw std::runtime_error(
            "Frank weighted PIC import found nonpositive represented mass.");
    }

    std::vector<std::size_t> source_order(particles.size());
    std::iota(source_order.begin(), source_order.end(), std::size_t{0});
    if (selected != particles.size()) {
        // SphDSMC checkpoints are normally already radius-sorted.  Sorting
        // indices here makes the documented reduced-sample rule explicit even
        // for a valid externally constructed v4/v5 restart.
        std::stable_sort(
            source_order.begin(),
            source_order.end(),
            [&](std::size_t first, std::size_t second) {
                return particles[first].r < particles[second].r;
            });
    }
    // The complete import keeps source ordering. A reduced equal-weight
    // sample uses evenly spaced ranks of the radial ordering so the
    // selection rule is reproducible for an unsorted valid source as well.
    auto selected_source_index = [&](std::size_t sample_index) {
        if (selected == particles.size()) {
            return sample_index;
        }
        const std::size_t rank = std::min(
            particles.size() - 1,
            (sample_index * particles.size() + particles.size() / 2)
                / selected);
        return source_order[rank];
    };

    long double selected_total_mass = 0.0L;
    for (std::size_t i = 0; i < selected; ++i) {
        selected_total_mass += static_cast<long double>(
            particles[selected_source_index(i)].mass);
    }
    if (!(selected_total_mass > 0.0L)) {
        throw std::runtime_error(
            "Frank weighted PIC import selected nonpositive represented mass.");
    }
    // Rescale only the explicitly reduced diagnostic sample. In a complete
    // import the selected and source totals agree, preserving all individual
    // weights as well as the phase-space mass distribution.
    const long double selected_mass_scale =
        source_total_mass / selected_total_mass;

    WeightedPICFrankRestartImport result;
    result.markers.resize(selected);
    for (std::size_t i = 0; i < selected; ++i) {
        const SphericalDSMCParticle& source =
            particles[selected_source_index(i)];
        WeightedPICMarker& marker = result.markers[i];
        marker.mass = static_cast<double>(
            static_cast<long double>(source.mass) * selected_mass_scale);
        marker.r = source.r;
        marker.w = source.vr;
        // The same specific angular momentum is stored in the canonical eta
        // coordinate. Radius, physical radial velocity, particle ID and epoch
        // retain their source meanings.
        marker.eta = 0.5 * source.L * source.L;
        marker.id = source.id;
    }
    result.time = header.time;
    result.step = header.step;
    return result;
}

}  // namespace ev
