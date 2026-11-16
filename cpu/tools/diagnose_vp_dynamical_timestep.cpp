
// Inspect which represented part of a Newtonian distribution controls its
// timestep. Input is a solver configuration or its declared restart. The tool
// reconstructs the same initial distribution and Poisson field, then reports
// activity-floor and hypothetical cell-mass-floor sensitivity. It prints
// retained mass and the limiting phase-space cell without transporting matter.
// The alternative floor table is a diagnostic comparison, not a changed
// production timestep policy.

#include "ev/solver.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    // ---- Parse and reconstruct the requested Vlasov--Poisson state --------
    if (argc != 2) {
        std::cerr << "Usage: diagnose_vp_dynamical_timestep CONFIG.ini\n";
        return 2;
    }
    try {
        ev::Params p;
        ev::load_config_file(argv[1], p);
        p.threads = 1;
        const ev::MpiContext mpi;
        ev::SolverState state = ev::create_solver_state(p, mpi);
        ev::initialize_distribution(state);
        // ---- Deposit moments and solve the instantaneous Poisson field ----
        const ev::Moments moments = ev::compute_global_vp_moments(
            state.f,
            state.r,
            state.w,
            state.L,
            state.wq,
            state.Lq,
            state.local_p,
            state.mpi);
        const ev::VlasovPoissonField field =
            ev::solve_vlasov_poisson_field(
                moments, state.r, state.t, state.global_p);
        // Use the reconstructed distribution maximum to define each relative
        // activity threshold. The potential is held fixed across all floor levels
        // so their differences isolate clock selection on this same state.
        const double maximum_f =
            *std::max_element(state.f.begin(), state.f.end());
        const std::vector<double> floors{
            1.0e-12,
            1.0e-11,
            1.0e-10,
            1.0e-9,
            1.0e-8,
            1.0e-7,
            1.0e-6,
        };
        std::cout << std::setprecision(17);
        std::cout
            << "active_floor_rel,dt,active_cells,active_phase_mass,"
            << "limiting_i,limiting_j,limiting_k,limiting_r,limiting_w,"
            << "limiting_L,limiting_f,limiting_cell_mass,"
            << "limiting_cell_mass_fraction,limiting_r_over_acceleration\n";
        const double total_mass = ev::vp_matter_mass(
            moments, state.r, state.global_p);
        // ---- Sensitivity to a distribution-function activity floor --------
        // This floor excludes numerically tiny phase-space tails from setting
        // a global step while reporting how much represented mass remains.
        for (double floor : floors) {
            ev::Params level = state.local_p;
            level.active_floor_abs = 0.0;
            level.active_floor_rel = floor;
            const double threshold = floor * maximum_f;
            long long active_cells = 0;
            long double active_phase_mass = 0.0L;
            double minimum_ratio = std::numeric_limits<double>::infinity();
            int limiting_i = -1;
            int limiting_j = -1;
            int limiting_k = -1;
            for (int i = 0; i < level.Nr; ++i) {
                for (int j = 0; j < level.Nw; ++j) {
                    for (int k = 0; k < level.NL; ++k) {
                        const double value =
                            state.f[ev::idx(i, j, k, level)];
                        if (value <= threshold) {
                            continue;
                        }
                        ++active_cells;
                        // The cell-average spherical measure is 8pi^2 dr dw deta. Reporting
                        // its product with f distinguishes a short clock carried by negligible
                        // mass from one carried by a substantial part of the distribution.
                        const double cell_measure =
                            8.0 * ev::PI * ev::PI
                            * state.kinetic_geometry.radial.width[
                                static_cast<std::size_t>(i)]
                            * state.kinetic_geometry.radial_velocity.width[
                                static_cast<std::size_t>(j)]
                            * state.kinetic_geometry.eta.width[
                                static_cast<std::size_t>(k)];
                        active_phase_mass +=
                            static_cast<long double>(cell_measure)
                            * static_cast<long double>(value);
                        const double radius =
                            state.r[static_cast<std::size_t>(i)];
                        const double acceleration =
                            state.L[static_cast<std::size_t>(k)]
                                * state.L[static_cast<std::size_t>(k)]
                                / (radius * radius * radius)
                            - field.phi_r[static_cast<std::size_t>(i)];
                        if (std::abs(acceleration) <= 1.0e-300) {
                            continue;
                        }
                        // The ratio r/|L^2/r^3-Phi_r| is the squared dynamical time used
                        // by this diagnostic. The output records the location attaining its
                        // minimum, separately from the complete configured choose_dt call.
                        const double ratio =
                            radius / std::abs(acceleration);
                        if (ratio < minimum_ratio) {
                            minimum_ratio = ratio;
                            limiting_i = i;
                            limiting_j = j;
                            limiting_k = k;
                        }
                    }
                }
            }
            const double dt = ev::choose_dt_vlasov_poisson(
                state.f,
                state.r,
                state.w,
                state.L,
                field,
                state.radial_step,
                state.min_dw,
                level,
                state.mpi);
            std::cout
                << floor << ','
                << dt << ','
                << active_cells << ','
                << static_cast<double>(active_phase_mass) << ','
                << limiting_i << ','
                << limiting_j << ','
                << limiting_k << ',';
            if (limiting_i >= 0) {
                std::cout
                    << state.r[static_cast<std::size_t>(limiting_i)] << ','
                    << state.w[static_cast<std::size_t>(limiting_j)] << ','
                    << state.L[static_cast<std::size_t>(limiting_k)] << ','
                    << state.f[ev::idx(
                           limiting_i, limiting_j, limiting_k, level)] << ',';
                const double limiting_cell_mass =
                    8.0 * ev::PI * ev::PI
                    * state.kinetic_geometry.radial.width[
                        static_cast<std::size_t>(limiting_i)]
                    * state.kinetic_geometry.radial_velocity.width[
                        static_cast<std::size_t>(limiting_j)]
                    * state.kinetic_geometry.eta.width[
                        static_cast<std::size_t>(limiting_k)]
                    * state.f[ev::idx(
                          limiting_i, limiting_j, limiting_k, level)];
                std::cout
                    << limiting_cell_mass << ','
                    << limiting_cell_mass / total_mass << ','
                    << minimum_ratio;
            } else {
                std::cout << "0,0,0,0,0,0,0";
            }
            std::cout << '\n';
        }
        // ---- Equivalent sensitivity expressed as cell mass fraction -------
        // The second table thresholds actual phase-space mass fraction rather
        // than f amplitude. It reports the hypothetical retained fraction and
        // dynamical estimate; it neither discards source mass nor advances it.
        const std::vector<double> cell_mass_floors{
            0.0,
            1.0e-18,
            1.0e-16,
            1.0e-14,
            1.0e-12,
            1.0e-10,
            1.0e-8,
        };
        std::cout
            << "cell_mass_floor_fraction,dt,retained_mass_fraction,"
            << "active_cells,limiting_i,limiting_j,limiting_k,"
            << "limiting_cell_mass_fraction\n";
        for (double floor : cell_mass_floors) {
            long long active_cells = 0;
            long double retained_mass = 0.0L;
            double minimum_ratio = std::numeric_limits<double>::infinity();
            double limiting_mass_fraction = 0.0;
            int limiting_i = -1;
            int limiting_j = -1;
            int limiting_k = -1;
            for (int i = 0; i < state.local_p.Nr; ++i) {
                for (int j = 0; j < state.local_p.Nw; ++j) {
                    for (int k = 0; k < state.local_p.NL; ++k) {
                        const double cell_mass =
                            8.0 * ev::PI * ev::PI
                            * state.kinetic_geometry.radial.width[
                                static_cast<std::size_t>(i)]
                            * state.kinetic_geometry.radial_velocity.width[
                                static_cast<std::size_t>(j)]
                            * state.kinetic_geometry.eta.width[
                                static_cast<std::size_t>(k)]
                            * state.f[ev::idx(i, j, k, state.local_p)];
                        const double mass_fraction = cell_mass / total_mass;
                        if (!(mass_fraction > floor)) {
                            continue;
                        }
                        ++active_cells;
                        retained_mass += cell_mass;
                        const double radius =
                            state.r[static_cast<std::size_t>(i)];
                        const double acceleration =
                            state.L[static_cast<std::size_t>(k)]
                                * state.L[static_cast<std::size_t>(k)]
                                / (radius * radius * radius)
                            - field.phi_r[static_cast<std::size_t>(i)];
                        if (std::abs(acceleration) <= 1.0e-300) {
                            continue;
                        }
                        const double ratio =
                            radius / std::abs(acceleration);
                        if (ratio < minimum_ratio) {
                            minimum_ratio = ratio;
                            limiting_i = i;
                            limiting_j = j;
                            limiting_k = k;
                            limiting_mass_fraction = mass_fraction;
                        }
                    }
                }
            }
            const double dt = std::isfinite(minimum_ratio)
                ? std::min(
                      state.local_p.dt_max,
                      std::sqrt(
                          2.0 * state.local_p.vp_dynamical_eta
                          * minimum_ratio))
                : state.local_p.dt_max;
            std::cout
                << floor << ','
                << dt << ','
                << static_cast<double>(retained_mass) / total_mass << ','
                << active_cells << ','
                << limiting_i << ','
                << limiting_j << ','
                << limiting_k << ','
                << limiting_mass_fraction << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "diagnose_vp_dynamical_timestep: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}
