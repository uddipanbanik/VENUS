
// Read a complete canonical Boltzmann--Poisson checkpoint and report its
// orbit-aligned distribution and collision carriers. The optional second
// argument selects one eta=L^2/2 plane for detailed output. This command
// checks binary record validity, then counts energy-moment support violations;
// those reported counts are diagnostics and do not advance or repair the state.

#include "ev/canonical_boltzmann_poisson_restart.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>

int main(int argc, char** argv) {
    // ---- Decode the complete checkpoint without advancing it -------------
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: inspect_canonical_restart RESTART [ETA_PLANE]\n";
        return 2;
    }
    const ev::CanonicalBoltzmannPoissonEventState state =
        ev::read_canonical_boltzmann_poisson_restart(
            std::filesystem::path(argv[1]));
    // ---- Audit orbit-atlas topology and stored energy moments ------------
    std::size_t maximum_eulerian_cell = 0;
    for (const ev::CanonicalOverlap& overlap : state.atlas.overlap) {
        maximum_eulerian_cell = std::max(
            maximum_eulerian_cell,
            static_cast<std::size_t>(overlap.eulerian_cell));
    }
    std::size_t invalid_mean_cells = 0;
    std::size_t invalid_variance_cells = 0;
    std::size_t invalid_second_upper_cells = 0;
    std::size_t worst_row = 0;
    std::size_t worst_phase_cell = 0;
    double worst_normalized_mean_excursion = 0.0;
    double worst_mean = 0.0;
    double worst_left = 0.0;
    double worst_right = 0.0;
    double worst_normalized_second_upper_excursion = 0.0;
    std::size_t worst_second_row = 0;
    std::size_t worst_second_phase_cell = 0;
    double worst_second_moment = 0.0;
    double worst_second_upper = 0.0;
    for (std::size_t row_index = 0;
         row_index < state.atlas.row.size();
         ++row_index) {
        const ev::CanonicalOrbitRow& row = state.atlas.row[row_index];
        const double span = row.energy_right - row.energy_left;
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            const double mass = row.phase_mass[q];
            if (!(mass > 0.0)) {
                continue;
            }
            // Positive mass in an energy interval requires its mean energy to lie
            // within that interval. Report excursions with an energy-scale roundoff
            // allowance and also normalize them by the actual interval span.
            const double mean = row.phase_energy_mass[q] / mass;
            const double lower_excursion = row.energy_left - mean;
            const double upper_excursion = mean - row.energy_right;
            const double excursion = std::max({0.0, lower_excursion, upper_excursion});
            const double scale = std::max({
                1.0,
                std::abs(row.energy_left),
                std::abs(row.energy_right),
                std::abs(mean),
            });
            const double tolerance = 8192.0
                * std::numeric_limits<double>::epsilon() * scale;
            if (excursion > tolerance) {
                ++invalid_mean_cells;
            }
            const double normalized = span > 0.0 ? excursion / span
                                                  : std::numeric_limits<double>::infinity();
            if (normalized > worst_normalized_mean_excursion) {
                worst_normalized_mean_excursion = normalized;
                worst_row = row_index;
                worst_phase_cell = q;
                worst_mean = mean;
                worst_left = row.energy_left;
                worst_right = row.energy_right;
            }
            // A positive distribution has M0*M2-M1^2>=0. Evaluate this variance
            // numerator in extended precision to separate cancellation from a
            // resolved violation of the stored moments.
            const long double variance_numerator =
                static_cast<long double>(row.phase_energy2_mass[q]) * mass
                - static_cast<long double>(row.phase_energy_mass[q])
                    * row.phase_energy_mass[q];
            const long double variance_scale = std::max(
                1.0L,
                std::abs(static_cast<long double>(row.phase_energy2_mass[q])
                    * mass));
            if (variance_numerator < -8192.0L
                    * std::numeric_limits<double>::epsilon()
                    * variance_scale) {
                ++invalid_variance_cells;
            }
            const long double mean2 = row.phase_energy2_mass[q] / mass;
            // For E in [a,b], E^2<=(a+b)E-a*b. This supplies an upper support
            // bound on the second moment in addition to nonnegative variance.
            const long double secant_upper =
                (static_cast<long double>(row.energy_left)
                 + static_cast<long double>(row.energy_right)) * mean
                - static_cast<long double>(row.energy_left)
                    * static_cast<long double>(row.energy_right);
            const long double second_scale = std::max({
                1.0L, std::abs(mean2), std::abs(secant_upper)});
            const long double second_tolerance = 16384.0L
                * std::numeric_limits<double>::epsilon() * second_scale;
            const long double second_excursion = std::max(
                0.0L, mean2 - secant_upper);
            if (second_excursion > second_tolerance) {
                ++invalid_second_upper_cells;
            }
            const double normalized_second = span > 0.0
                ? static_cast<double>(second_excursion)
                    / (span * span)
                : std::numeric_limits<double>::infinity();
            if (normalized_second > worst_normalized_second_upper_excursion) {
                worst_normalized_second_upper_excursion = normalized_second;
                worst_second_row = row_index;
                worst_second_phase_cell = q;
                worst_second_moment = static_cast<double>(mean2);
                worst_second_upper = static_cast<double>(secant_upper);
            }
        }
    }
    // ---- Report global state, represented mass, and moment bounds --------
    std::cout << "time=" << state.time
              << " step=" << state.step
              << " eulerian_cells=" << state.eulerian_f.size()
              << " maximum_overlap_cell=" << maximum_eulerian_cell
              << " rows=" << state.atlas.row.size()
              << " columns=" << state.atlas.column.size()
              << " overlaps=" << state.atlas.overlap.size()
              << " eta_planes=" << state.atlas.eta_plane_coordinate.size()
              << " open_markers=" << state.collision.open_marker.size()
              << " detached_markers="
              << state.collision.detached_marker.size()
              << " signed_positive_markers="
              << state.collision.signed_increment_positive.size()
              << " signed_negative_markers="
              << state.collision.signed_increment_negative.size()
              << " represented_mass="
              << 8.0 * ev::PI * ev::PI
                    * ev::canonical_orbit_atlas_mass(state.atlas)
                    + ev::canonical_total_carrier_mass(state.collision)
              << " invalid_mean_cells=" << invalid_mean_cells
              << " invalid_variance_cells=" << invalid_variance_cells
              << " invalid_second_upper_cells="
              << invalid_second_upper_cells
              << " worst_mean_excursion_rows="
              << worst_normalized_mean_excursion
              << " worst_row=" << worst_row
              << " worst_phase_cell=" << worst_phase_cell
              << " worst_mean=" << worst_mean
              << " worst_energy_left=" << worst_left
              << " worst_energy_right=" << worst_right
              << " worst_second_excursion_rows2="
              << worst_normalized_second_upper_excursion
              << " worst_second_row=" << worst_second_row
              << " worst_second_phase_cell=" << worst_second_phase_cell
              << " worst_second_moment=" << worst_second_moment
              << " worst_second_upper=" << worst_second_upper
              << '\n';
    if (argc == 3) {
        // ---- Optional detailed inventory for one eta plane ----------------
        const int plane = std::stoi(argv[2]);
        if (plane < 0
            || plane + 1
                >= static_cast<int>(state.atlas.eta_plane_row_offset.size())) {
            std::cerr << "ETA_PLANE is outside the restart atlas.\n";
            return 2;
        }
        const std::size_t begin = state.atlas.eta_plane_row_offset[
            static_cast<std::size_t>(plane)];
        const std::size_t end = state.atlas.eta_plane_row_offset[
            static_cast<std::size_t>(plane + 1)];
        // The atlas stores canonical dr dw deta mass. Multiplication by
        // 8pi^2 restores the spherical physical phase-space normalization
        // when reporting each eta-plane mass.
        constexpr long double physical_factor =
            8.0L * static_cast<long double>(ev::PI)
            * static_cast<long double>(ev::PI);
        long double plane_mass = 0.0L;
        for (std::size_t row_index = begin; row_index < end; ++row_index) {
            const ev::CanonicalOrbitRow& row = state.atlas.row[row_index];
            long double row_mass = 0.0L;
            for (double mass : row.phase_mass) {
                row_mass += mass;
            }
            plane_mass += row_mass;
            std::cout << "eta_plane=" << plane
                      << " eta=" << row.eta
                      << " local_row=" << row_index - begin
                      << " global_row=" << row_index
                      << " action_left=" << row.action_left
                      << " action_right=" << row.action_right
                      << " energy_left=" << row.energy_left
                      << " energy_right=" << row.energy_right
                      << " physical_mass="
                      << static_cast<double>(physical_factor * row_mass)
                      << '\n';
        }
        std::cout << "eta_plane=" << plane
                  << " physical_plane_mass="
                  << static_cast<double>(physical_factor * plane_mass)
                  << '\n';
    }
    return 0;
}
