
// Split each numerical weighted-particle sample into equal-weight clones at
// the same saved phase-space point. Inputs are a complete checkpoint, output
// path and split factor 2..16. Output is a new particle initialization with the
// same represented phase-space mass and epoch, and a newly initialized field
// schedule. This changes sampling and differs from continuing the exact old
// stochastic trajectory or applying a physical collision.

#include "ev/weighted_vlasov_pic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t parse_factor(const char* text) {
    const unsigned long long value = std::stoull(text);
    if (value < 2 || value > 16) {
        throw std::runtime_error("split factor must be in [2, 16]");
    }
    return static_cast<std::size_t>(value);
}

long double total_mass(const std::vector<ev::WeightedPICMarker>& markers) {
    long double result = 0.0L;
    for (const auto& marker : markers) {
        result += static_cast<long double>(marker.mass);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        // ---- Parse, decode, and validate identifier capacity --------------
        if (argc != 4 && argc != 5) {
            throw std::runtime_error(
                "usage: refine_weighted_pic_restart INPUT OUTPUT SPLIT_FACTOR "
                "[--reinitialize-field]");
        }
        const bool reinitialize_field = argc == 5
            && std::string(argv[4]) == "--reinitialize-field";
        if (argc == 5 && !reinitialize_field) {
            throw std::runtime_error("unknown option: " + std::string(argv[4]));
        }
        const std::filesystem::path input = argv[1];
        const std::filesystem::path output = argv[2];
        const std::size_t factor = parse_factor(argv[3]);
        ev::WeightedPICRestartState state = ev::read_weighted_pic_restart(input);
        if (state.markers.empty()) {
            throw std::runtime_error("cannot refine an empty restart");
        }
        if (state.has_frozen_field && !reinitialize_field) {
            throw std::runtime_error(
                "refining a frozen-field checkpoint requires "
                "--reinitialize-field; the output starts a new field schedule");
        }
        // A pending collision batch owns its full carried interval and sampling
        // state. Complete that interval before changing the numerical population.
        if (state.collision_event_age != 0.0) {
            throw std::runtime_error(
                "a pending collision interval must finish before marker "
                "refinement; select a checkpoint with collision_event_age=0");
        }
        const std::size_t original_count = state.markers.size();
        if (original_count > std::numeric_limits<std::size_t>::max() / factor) {
            throw std::runtime_error("refined marker count overflows size_t");
        }
        std::uint64_t next_id = 0;
        for (const auto& marker : state.markers) {
            // Subhalo internal dispersion and energy describe a physical object.
            // Cloning its physical mass would change its encounter law, so this
            // sampling operation admits only markers with zero such internal state.
            if (marker.internal_energy != 0.0
                || marker.internal_velocity_dispersion != 0.0) {
                throw std::runtime_error(
                    "refinement requires zero internal energy and internal "
                    "velocity dispersion: splitting subhalo objects "
                    "changes their physical masses and encounter model");
            }
            if (marker.id == std::numeric_limits<std::uint64_t>::max()) {
                throw std::runtime_error("marker ID space is exhausted");
            }
            next_id = std::max(next_id, marker.id + 1);
        }
        const std::size_t clones = original_count * (factor - 1);
        if (clones > std::numeric_limits<std::uint64_t>::max() - next_id) {
            throw std::runtime_error("refined marker IDs overflow uint64_t");
        }
        // ---- Replace each numerical sample by equal-weight clones ---------
        // Clones retain the same mass-weighted phase-space moments. Material
        // SIDM interprets each clone as a numerical sample with a smaller
        // represented mass; subhalo internal state was rejected above.
        const long double mass_before = total_mass(state.markers);
        std::vector<ev::WeightedPICMarker> refined;
        refined.reserve(original_count * factor);
        // Each original packet becomes factor colocated samples with weight
        // W/factor. Keep the original identity on one sample and assign fresh
        // identities to the rest, preserving the pointwise weighted distribution.
        for (auto marker : state.markers) {
            marker.mass /= static_cast<double>(factor);
            refined.push_back(marker);
            for (std::size_t copy = 1; copy < factor; ++copy) {
                marker.id = next_id++;
                refined.push_back(marker);
            }
        }
        state.markers.swap(refined);
        const long double mass_after = total_mass(state.markers);
        // Audit the sum of the actual stored double weights after splitting.
        // The check detects accumulated representation error before writing
        // the new initialization record.
        const long double relative_mass_error = std::abs(mass_after - mass_before)
            / std::max(std::abs(mass_before), std::numeric_limits<long double>::min());
        if (relative_mass_error > 64.0L * std::numeric_limits<double>::epsilon()) {
            throw std::runtime_error("marker refinement failed mass-conservation audit");
        }
        // The marker population affects field/collision estimates. Publish
        // particle initialization (v6), with fresh estimates and no cached
        // frozen field, so the receiving solver explicitly initializes them.
        state.has_frozen_field = false;
        state.frozen_field = {};
        state.frozen_field_contract.clear();
        state.potential_event_age = 0.0;
        state.collision_timescale_estimate =
            std::numeric_limits<double>::infinity();
        state.potential_timescale_estimate =
            std::numeric_limits<double>::infinity();
        ev::write_weighted_pic_restart(output, state);
        std::cout << "input_markers=" << original_count
                  << " output_markers=" << state.markers.size()
                  << " factor=" << factor
                  << " step=" << state.step
                  << " time=" << state.time
                  << " relative_mass_error=" << static_cast<double>(relative_mass_error)
                  << " output_kind=particle_initialization_v6"
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
