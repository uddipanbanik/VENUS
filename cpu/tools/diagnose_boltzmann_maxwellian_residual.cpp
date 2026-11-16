
// Measure how closely the finite-angle Boltzmann discretization preserves a
// Maxwellian equilibrium. Input supplies a binary carrier velocity grid and
// angular quadrature counts; its original distribution is not evolved here.
// The tool constructs a normalized Maxwellian in selected radial shells and
// applies one small collision interval with and without moment repair. Output
// separates distribution/pressure residuals, invariant errors and skipped
// collision activity, normalized by the corresponding collision rate.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ev/finite_angle_boltzmann.hpp"

namespace {

template <typename T>
T read_binary(std::ifstream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("Unexpected end of initial-data file.");
    }
    return value;
}

struct CarrierGrid {
    std::vector<double> r;
    std::vector<double> w;
    std::vector<double> L;
};

// Read only the EVICRWL1 grid needed for this independent velocity-space
// equilibrium experiment. Its saved distribution values play no role
// in the constructed Maxwellian below.
CarrierGrid read_carrier_grid(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open initial-data file: " + path);
    }
    char magic[16]{};
    input.read(magic, sizeof(magic));
    const std::int32_t version = read_binary<std::int32_t>(input);
    const std::int32_t nr = read_binary<std::int32_t>(input);
    const std::int32_t nw = read_binary<std::int32_t>(input);
    const std::int32_t nl = read_binary<std::int32_t>(input);
    if (std::string(magic, magic + 8) != "EVICRWL1" || version != 1
        || nr < 2 || nw < 2 || nl < 2) {
        throw std::runtime_error("Unsupported EVICRWL1 carrier.");
    }
    CarrierGrid grid;
    grid.r.resize(static_cast<std::size_t>(nr));
    grid.w.resize(static_cast<std::size_t>(nw));
    grid.L.resize(static_cast<std::size_t>(nl));
    input.read(reinterpret_cast<char*>(grid.r.data()), nr * sizeof(double));
    input.read(reinterpret_cast<char*>(grid.w.data()), nw * sizeof(double));
    input.read(reinterpret_cast<char*>(grid.L.data()), nl * sizeof(double));
    if (!input) {
        throw std::runtime_error("Truncated EVICRWL1 carrier grid.");
    }
    return grid;
}

double nearest_radius(const std::vector<double>& r, double requested) {
    return *std::min_element(r.begin(), r.end(), [&](double a, double b) {
        return std::abs(a - requested) < std::abs(b - requested);
    });
}

struct ResidualResult {
    std::size_t active_cells = 0;
    double max_loss_rate = 0.0;
    double relative_l1_rate = 0.0;
    double rate_per_mean_collision_frequency = 0.0;
    double attempted_event_rate = 0.0;
    double attempted_to_exact_event_rate = 0.0;
    double radial_pressure_rate_per_collision_frequency = 0.0;
    double tangential_pressure_rate_per_collision_frequency = 0.0;
    double relative_mass_error = 0.0;
    double relative_momentum_error = 0.0;
    double relative_energy_error = 0.0;
    double skipped_fraction = 0.0;
};

ResidualResult maxwellian_residual(
    double radius,
    double temperature,
    const CarrierGrid& grid,
    int pair_phi_points,
    int scatter_mu_points,
    int scatter_phi_points,
    int threads,
    bool repair_moments) {
    // Build a one-shell collision problem.  A Maxwellian is an equilibrium
    // (null mode) of elastic Boltzmann scattering, so any change measures
    // velocity-grid, angular-quadrature, or projection error.
    ev::Params p;
    p.Nr = 1;
    p.Nw = static_cast<int>(grid.w.size());
    p.NL = static_cast<int>(grid.L.size());
    p.threads = threads;
    p.bp_cross_section = "constant";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 1.0;
    p.bp_boltzmann_pair_phi_points = pair_phi_points;
    p.bp_boltzmann_scatter_mu_points = scatter_mu_points;
    p.bp_boltzmann_scatter_phi_points = scatter_phi_points;
    p.bp_boltzmann_mass_tol = 1.0e-9;
    p.bp_boltzmann_conserve_moments = repair_moments;

    const std::vector<double> r{radius};
    const std::vector<double> wq = ev::quadrature_weights(grid.w);
    const std::vector<double> Lq = ev::quadrature_weights(grid.L);
    const std::vector<double> volume =
        ev::bp_fa_shell_velocity_volumes(radius, wq, grid.L, Lq, p);
    std::vector<double> f(static_cast<std::size_t>(p.Nw * p.NL), 0.0);
    for (int j = 0; j < p.Nw; ++j) {
        for (int k = 0; k < p.NL; ++k) {
            const double vt = grid.L[static_cast<std::size_t>(k)] / radius;
            const double speed2 =
                grid.w[static_cast<std::size_t>(j)] * grid.w[static_cast<std::size_t>(j)]
                + vt * vt;
            f[ev::idx(0, j, k, p)] = std::exp(-0.5 * speed2 / temperature);
        }
    }
    const ev::BPFAShellMoments unnormalized =
        ev::bp_fa_shell_moments(f, 0, radius, grid.w, grid.L, volume, p);
    for (double& value : f) {
        value /= unnormalized.mass;
    }

    const std::vector<ev::BPFAVelocityCell> active =
        ev::bp_fa_active_cells(f, 0, radius, grid.w, grid.L, volume, p);
    const double max_loss_rate =
        ev::bp_fa_constant_shell_loss_rate_upper_bound(active, p);
    // Choose a small dimensionless exposure relative to the maximum loss
    // rate. Residual divided by dt measures an approximate null-operator
    // error at fixed velocity and angular resolution.
    const double dt = 0.02 / std::max(max_loss_rate, 1.0e-300);
    const std::vector<double> initial = f;
    const ev::BPFAShellMoments before =
        ev::bp_fa_shell_moments(f, 0, radius, grid.w, grid.L, volume, p);
    auto pressures = [&](const std::vector<double>& distribution) {
        double radial = 0.0;
        double tangential = 0.0;
        const double mean_w = before.radial_momentum / before.mass;
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const int shell_index = j * p.NL + k;
                const double n = distribution[ev::idx(0, j, k, p)]
                    * volume[static_cast<std::size_t>(shell_index)];
                const double dw = grid.w[static_cast<std::size_t>(j)] - mean_w;
                const double vt = grid.L[static_cast<std::size_t>(k)] / radius;
                radial += n * dw * dw;
                tangential += n * vt * vt;
            }
        }
        return std::array<double, 2>{radial, tangential};
    };
    const std::array<double, 2> pressure_before = pressures(f);
    // Apply one deliberately small finite-angle collision step.  "Moment
    // repair" projects roundoff/remap defects back onto the exactly conserved
    // mass, radial momentum, and kinetic-energy moments.
    const ev::BPFARemapBudget budget = ev::bp_fa_apply_shell_substep(
        f, 0, dt, r, grid.w, grid.L, wq, Lq, p, threads);
    const ev::BPFAShellMoments after =
        ev::bp_fa_shell_moments(f, 0, radius, grid.w, grid.L, volume, p);
    const std::array<double, 2> pressure_after = pressures(f);

    double weighted_l1 = 0.0;
    for (int j = 0; j < p.Nw; ++j) {
        for (int k = 0; k < p.NL; ++k) {
            const int shell_index = j * p.NL + k;
            weighted_l1 += volume[static_cast<std::size_t>(shell_index)]
                * std::abs(f[ev::idx(0, j, k, p)] - initial[ev::idx(0, j, k, p)]);
        }
    }
    // For a unit-mass Maxwellian with one-dimensional variance T, the
    // mean pair relative speed is 4sqrt(T/pi). The pair event density has
    // the additional factor 1/2 for unordered incoming partners.
    const double mean_relative_speed = 4.0 * std::sqrt(temperature / ev::PI);
    ResidualResult result;
    result.active_cells = active.size();
    result.max_loss_rate = max_loss_rate;
    result.relative_l1_rate = weighted_l1 / (before.mass * dt);
    result.rate_per_mean_collision_frequency =
        result.relative_l1_rate / (before.mass * p.bp_sigma0 * mean_relative_speed);
    result.attempted_event_rate = budget.attempted_event_density / dt;
    const double exact_event_rate =
        0.5 * before.mass * before.mass * p.bp_sigma0 * mean_relative_speed;
    result.attempted_to_exact_event_rate =
        result.attempted_event_rate / exact_event_rate;
    const double mean_collision_frequency =
        before.mass * p.bp_sigma0 * mean_relative_speed;
    result.radial_pressure_rate_per_collision_frequency =
        (pressure_after[0] - pressure_before[0])
        / (pressure_before[0] * dt * mean_collision_frequency);
    result.tangential_pressure_rate_per_collision_frequency =
        (pressure_after[1] - pressure_before[1])
        / (pressure_before[1] * dt * mean_collision_frequency);
    // Report conserved-moment errors alongside the full-distribution
    // residual. Moment repair can restore invariants while a velocity-grid
    // or angular-remap equilibrium defect remains visible in f.
    result.relative_mass_error = (after.mass - before.mass) / before.mass;
    result.relative_momentum_error =
        (after.radial_momentum - before.radial_momentum) / before.mass;
    result.relative_energy_error =
        (after.kinetic_energy - before.kinetic_energy) / before.kinetic_energy;
    result.skipped_fraction = budget.attempted_event_density > 0.0
        ? budget.skipped_event_density / budget.attempted_event_density
        : 0.0;
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    // ---- Parse grid and angular-quadrature controls ----------------------
    if (argc < 2 || argc > 6) {
        std::cerr
            << "Usage: diagnose_boltzmann_maxwellian_residual IC_FILE"
            << " [pair_phi=4] [scatter_mu=2] [scatter_phi=4] [threads=1]\n";
        return 2;
    }
    const std::string path = argv[1];
    const int pair_phi = (argc > 2) ? std::stoi(argv[2]) : 4;
    const int scatter_mu = (argc > 3) ? std::stoi(argv[3]) : 2;
    const int scatter_phi = (argc > 4) ? std::stoi(argv[4]) : 4;
    const int threads = (argc > 5) ? std::stoi(argv[5]) : 1;
    const CarrierGrid grid = read_carrier_grid(path);
    constexpr double temperature = 0.09;
    const std::vector<double> requested_radii{0.019, 0.047, 0.105, 0.206, 1.039};

    // ---- Compare raw and conservative collision projections -------------
    std::cout << std::setprecision(10)
              << "# Nw=" << grid.w.size()
              << " NL=" << grid.L.size()
              << " T=" << temperature
              << " quadrature=" << pair_phi << 'x' << scatter_mu << 'x' << scatter_phi
              << " threads=" << threads << '\n'
              << "# columns: r repair active max_loss l1_rate"
              << " l1_rate_over_mean_collision_frequency attempted_event_rate"
              << " attempted_to_exact_event_rate dPr_over_Pr_per_collision"
              << " dPt_over_Pt_per_collision rel_dM dP_over_M rel_dK skipped\n";
    for (double requested : requested_radii) {
        const double radius = nearest_radius(grid.r, requested);
        for (bool repair : {false, true}) {
            const ResidualResult result = maxwellian_residual(
                radius,
                temperature,
                grid,
                pair_phi,
                scatter_mu,
                scatter_phi,
                threads,
                repair);
            std::cout << radius << ' '
                      << static_cast<int>(repair) << ' '
                      << result.active_cells << ' '
                      << result.max_loss_rate << ' '
                      << result.relative_l1_rate << ' '
                      << result.rate_per_mean_collision_frequency << ' '
                      << result.attempted_event_rate << ' '
                      << result.attempted_to_exact_event_rate << ' '
                      << result.radial_pressure_rate_per_collision_frequency << ' '
                      << result.tangential_pressure_rate_per_collision_frequency << ' '
                      << result.relative_mass_error << ' '
                      << result.relative_momentum_error << ' '
                      << result.relative_energy_error << ' '
                      << result.skipped_fraction << '\n';
        }
    }
    return 0;
}
