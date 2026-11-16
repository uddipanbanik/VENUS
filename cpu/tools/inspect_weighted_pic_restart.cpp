
// Inspect the complete saved state of a Newtonian weighted-particle run.
// The printed inventory includes physical epoch, marker lifecycle states,
// represented mass, internal energy, effective sample size and the carried
// collision age/random counter. Inspection leaves the source file unchanged.
// An explicit particle export instead initializes a new field/event schedule
// at the saved epoch and is admitted only at a completed collision interval.

#include "ev/weighted_vlasov_pic.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        // ---- Command line and checkpoint decoding ------------------------
        if (argc != 2 && argc != 4) {
            throw std::runtime_error(
                "usage: inspect_weighted_pic_restart RESTART_STATE "
                "[--export-particles OUTPUT]");
        }
        const bool export_particles = argc == 4
            && std::string(argv[2]) == "--export-particles";
        if (argc == 4 && !export_particles) {
            throw std::runtime_error("unknown option: " + std::string(argv[2]));
        }
        ev::WeightedPICRestartState state =
            ev::read_weighted_pic_restart(std::filesystem::path(argv[1]));
        // A particle-only initialization has no unfinished frozen collision
        // interval. Require zero carried collision age before discarding the
        // cached field and reconstructing a new schedule.
        if (export_particles && state.collision_event_age != 0.0) {
            throw std::runtime_error(
                "a pending collision interval must finish before particle "
                "export; select a checkpoint with collision_event_age=0");
        }

        // ---- Physical-state inventory -----------------------------------
        // Marker mass includes all lifecycle states because delayed or
        // escaped packets still belong to the solver's conservation ledger.
        // The reported active_neff=(sum m)^2/sum(m^2) is the effective
        // equal-weight sample size of an unequal-mass active population.
        long double mass = 0.0L;
        long double internal_energy = 0.0L;
        long double active_mass = 0.0L;
        long double active_squared_mass = 0.0L;
        long double active_mass_sigma = 0.0L;
        long double active_mass_sigma2 = 0.0L;
        long double active_mass_w_sigma = 0.0L;
        double active_minimum_mass = std::numeric_limits<double>::infinity();
        double active_maximum_mass = 0.0;
        std::size_t active = 0;
        std::size_t delayed = 0;
        std::size_t escaped = 0;
        for (const ev::WeightedPICMarker& marker : state.markers) {
            mass += marker.mass;
            internal_energy += marker.internal_energy;
            if (marker.state == ev::WeightedPICMarkerState::active) {
                ++active;
                active_mass += marker.mass;
                active_squared_mass +=
                    static_cast<long double>(marker.mass) * marker.mass;
                active_mass_sigma += marker.mass
                    * marker.internal_velocity_dispersion;
                active_mass_sigma2 += marker.mass
                    * marker.internal_velocity_dispersion
                    * marker.internal_velocity_dispersion;
                active_mass_w_sigma += marker.mass * marker.w
                    * marker.internal_velocity_dispersion;
                active_minimum_mass = std::min(
                    active_minimum_mass, marker.mass);
                active_maximum_mass = std::max(
                    active_maximum_mass, marker.mass);
            } else if (
                marker.state == ev::WeightedPICMarkerState::delayed_inner) {
                ++delayed;
            } else {
                ++escaped;
            }
        }

        // ---- Human-readable restart provenance --------------------------
        std::cout << std::setprecision(17)
                  << "time=" << state.time
                  << " step=" << state.step
                  << " markers=" << state.markers.size()
                  << " active=" << active
                  << " delayed=" << delayed
                  << " escaped=" << escaped
                  << " mass=" << static_cast<double>(mass)
                  << " internal_energy="
                  << static_cast<double>(internal_energy)
                  << " active_min_mass="
                  << (std::isfinite(active_minimum_mass)
                          ? active_minimum_mass : 0.0)
                  << " active_max_mass=" << active_maximum_mass
                  << " active_neff="
                  << (active_squared_mass > 0.0L
                          ? static_cast<double>(
                                active_mass * active_mass
                                / active_squared_mass)
                          : 0.0)
                  << " active_mass_sigma="
                  << static_cast<double>(active_mass_sigma)
                  << " active_mass_sigma2="
                  << static_cast<double>(active_mass_sigma2)
                  << " active_mass_w_sigma="
                  << static_cast<double>(active_mass_w_sigma)
                  << " collision_event_age=" << state.collision_event_age
                  << " collision_counter="
                  << state.collision_sampler.next_counter
                  << " frozen_field=" << state.has_frozen_field
                  << '\n';
        // Export is an explicit state conversion, separate from exact native
        // continuation. A distinct path protects the input and the code below
        // clears only field/schedule data while retaining the particle record.
        if (export_particles) {
            const auto input = std::filesystem::path(argv[1]);
            const auto output = std::filesystem::path(argv[3]);
            if (std::filesystem::absolute(input).lexically_normal()
                    == std::filesystem::absolute(output).lexically_normal()
                || (std::filesystem::exists(output)
                    && std::filesystem::equivalent(input, output))) {
                throw std::runtime_error("particle export requires a distinct output path");
            }
            // Keep particles, identifiers, RNG positions, time, and step.
            // A receiving CPU/GPU solver constructs its own field and clocks.
            state.has_frozen_field = false;
            state.frozen_field = {};
            state.frozen_field_contract.clear();
            state.potential_event_age = 0.0;
            state.collision_event_age = 0.0;
            state.collision_timescale_estimate =
                std::numeric_limits<double>::infinity();
            state.potential_timescale_estimate =
                std::numeric_limits<double>::infinity();
            ev::write_weighted_pic_restart(output, state);
            std::cout << "output_kind=particle_initialization_v6"
                      << " field_and_event_schedule=reinitialize"
                      << " output=" << output.string() << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
