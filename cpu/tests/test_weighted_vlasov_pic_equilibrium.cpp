// Weighted-PIC equilibrium/noise tests: quiet sampling, self-consistent
// transport drift, density error, and marker-count convergence.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ev/weighted_vlasov_pic.hpp"

namespace {

constexpr double plummer_df_normalization =
    24.0 * 1.4142135623730950488 / (7.0 * ev::PI * ev::PI * ev::PI);

double plummer_relative_energy(double radius, double radial_velocity, double eta) {
    return 1.0 / std::sqrt(1.0 + radius * radius)
        - 0.5 * radial_velocity * radial_velocity - eta / (radius * radius);
}

double plummer_df(double radius, double radial_velocity, double eta) {
    const double relative_energy =
        plummer_relative_energy(radius, radial_velocity, eta);
    return relative_energy > 0.0
        ? plummer_df_normalization * std::pow(relative_energy, 3.5)
        : 0.0;
}

double maximum_radial_relative_potential(
    double r_lower,
    double r_upper,
    double eta_lower) {
    auto value = [eta_lower](double radius) {
        return 1.0 / std::sqrt(1.0 + radius * radius)
            - eta_lower / (radius * radius);
    };
    auto derivative = [eta_lower](double radius) {
        return -radius / std::pow(1.0 + radius * radius, 1.5)
            + 2.0 * eta_lower / (radius * radius * radius);
    };

    double maximum = std::max(value(r_lower), value(r_upper));
    if (derivative(r_lower) > 0.0 && derivative(r_upper) < 0.0) {
        double lower = r_lower;
        double upper = r_upper;
        for (int iteration = 0; iteration < 80; ++iteration) {
            const double midpoint = 0.5 * (lower + upper);
            if (derivative(midpoint) > 0.0) {
                lower = midpoint;
            } else {
                upper = midpoint;
            }
        }
        maximum = std::max(maximum, value(0.5 * (lower + upper)));
    }
    return maximum;
}

double plummer_cell_upper_bound(
    double r_lower,
    double r_upper,
    double w_lower,
    double w_upper,
    double eta_lower,
    double eta_upper) {
    (void)eta_upper;
    const double minimum_abs_w = w_lower <= 0.0 && w_upper >= 0.0
        ? 0.0
        : std::min(std::abs(w_lower), std::abs(w_upper));
    const double relative_energy = maximum_radial_relative_potential(
            r_lower, r_upper, eta_lower)
        - 0.5 * minimum_abs_w * minimum_abs_w;
    if (!(relative_energy > 0.0)) {
        return 0.0;
    }
    const double bound =
        plummer_df_normalization * std::pow(relative_energy, 3.5);
    return std::nextafter(
        bound * (1.0 + 1024.0 * std::numeric_limits<double>::epsilon()),
        std::numeric_limits<double>::infinity());
}

ev::Params equilibrium_params(int radial_cells) {
    ev::Params p;
    p.physics_model = "vlasov_poisson";
    p.vp_universe = "static";
    p.vp_gravity = 1.0;
    p.vp_interpolation = "hamiltonian_sldg";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "power";
    p.kinetic_eta_power = 3.0;
    p.Nr = radial_cells;
    p.Nw = 20;
    p.NL = 16;
    p.r_min = 0.03;
    p.r_max = 20.0;
    p.radial_grid = "log";
    p.w_max = 1.5;
    p.L_max = 7.0;
    p.inner_boundary = "reflecting";
    p.outer_boundary = "reflecting";
    p.vp_orbit_regularization = "off";
    p.integrator = "semi_lagrangian_kick_drift_kick";
    return p;
}

std::vector<double> plummer_cell_averages(
    const ev::PhaseSpaceCellGeometry& geometry,
    const ev::Params& p) {
    const auto& rule = ev::phase_space_gauss_legendre_rule(4);
    std::vector<double> result(
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 0.0);
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                long double integral = 0.0L;
                for (std::size_t qr = 0; qr < rule.first.size(); ++qr) {
                    const double radius =
                        geometry.radial.center[static_cast<std::size_t>(i)]
                        + 0.5 * geometry.radial.width[static_cast<std::size_t>(i)]
                            * rule.first[qr];
                    for (std::size_t qw = 0; qw < rule.first.size(); ++qw) {
                        const double radial_velocity =
                            geometry.radial_velocity.center[static_cast<std::size_t>(j)]
                            + 0.5
                                * geometry.radial_velocity.width[static_cast<std::size_t>(j)]
                                * rule.first[qw];
                        for (std::size_t qe = 0; qe < rule.first.size(); ++qe) {
                            const double eta =
                                geometry.eta.center[static_cast<std::size_t>(k)]
                                + 0.5 * geometry.eta.width[static_cast<std::size_t>(k)]
                                    * rule.first[qe];
                            integral += static_cast<long double>(
                                rule.second[qr] * rule.second[qw]
                                * rule.second[qe])
                                * plummer_df(radius, radial_velocity, eta);
                        }
                    }
                }
                result[ev::idx(i, j, k, p)] =
                    static_cast<double>(integral / 8.0L);
            }
        }
    }
    // The exact f(E) integral is even in w.  Average separately rounded
    // mirror quadratures so a support-boundary ulp cannot defeat quiet pairs.
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw / 2; ++j) {
            const int mirror = p.Nw - 1 - j;
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t negative = ev::idx(i, j, k, p);
                const std::size_t positive = ev::idx(i, mirror, k, p);
                const double symmetric = 0.5 * (result[negative] + result[positive]);
                result[negative] = symmetric;
                result[positive] = symmetric;
            }
        }
    }
    return result;
}

double shell_l1_difference(
    const std::vector<double>& first,
    const std::vector<double>& second,
    double normalization) {
    assert(first.size() == second.size());
    long double difference = 0.0L;
    for (std::size_t i = 0; i < first.size(); ++i) {
        difference += std::abs(
            static_cast<long double>(first[i]) - second[i]);
    }
    return static_cast<double>(difference / normalization);
}

double integrated_density_uncertainty(
    const ev::WeightedPICNoiseDiagnostics& noise,
    const ev::WeightedPICRadialMesh& mesh,
    double normalization) {
    long double uncertainty = 0.0L;
    for (std::size_t i = 0; i < mesh.shell_volume.size(); ++i) {
        uncertainty += static_cast<long double>(noise.density_uncertainty[i])
            * mesh.shell_volume[i];
    }
    return static_cast<double>(uncertainty / normalization);
}

double normalized_absolute_current(
    const ev::WeightedPICDepositedMoments& deposited,
    double normalization) {
    long double current = 0.0L;
    for (double shell_current : deposited.shell_radial_momentum) {
        current += std::abs(static_cast<long double>(shell_current));
    }
    return static_cast<double>(current / normalization);
}

double plummer_target_l1(
    const ev::WeightedPICDepositedMoments& deposited,
    const ev::WeightedPICRadialMesh& mesh) {
    auto enclosed_mass = [](double radius) {
        return radius * radius * radius
            / std::pow(1.0 + radius * radius, 1.5);
    };
    long double difference = 0.0L;
    long double target_mass = 0.0L;
    for (std::size_t i = 0; i < deposited.shell_mass.size(); ++i) {
        const double target = enclosed_mass(mesh.edge[i + 1])
            - enclosed_mass(mesh.edge[i]);
        difference += std::abs(
            static_cast<long double>(deposited.shell_mass[i]) - target);
        target_mass += target;
    }
    return static_cast<double>(difference / target_mass);
}

double discrete_total_energy(
    const ev::WeightedPICDepositedMoments& deposited,
    const ev::VlasovPoissonField& field) {
    assert(deposited.shell_mass.size() == field.phi.size());
    long double kinetic = 0.0L;
    long double potential = 0.0L;
    for (std::size_t i = 0; i < deposited.shell_mass.size(); ++i) {
        kinetic += 0.5L
            * (static_cast<long double>(
                   deposited.shell_radial_second_moment[i])
               + deposited.shell_tangential_second_moment[i]);
        potential += 0.5L
            * static_cast<long double>(deposited.shell_mass[i]) * field.phi[i];
    }
    return static_cast<double>(kinetic + potential);
}

double enclosed_mesh_mass(
    const ev::WeightedPICDepositedMoments& deposited,
    const ev::WeightedPICRadialMesh& mesh,
    double radius) {
    long double mass = 0.0L;
    for (std::size_t i = 0; i < deposited.shell_mass.size(); ++i) {
        if (mesh.volume_coordinate_center[i] <= radius * radius * radius / 3.0) {
            mass += deposited.shell_mass[i];
        }
    }
    return static_cast<double>(mass);
}

struct EquilibriumRunOptions {
    int steps = 100;
    int output_every = 100;
    double dt = 0.01;
    int markers_per_occupied_cell = 24;
    std::size_t marker_budget = 80000;
    int radial_cells = 24;
    int field_radial_cells = 24;
    unsigned int threads = 1;
    ev::WeightedPICParticleShape shape = ev::WeightedPICParticleShape::cic;
    std::filesystem::path snapshot_directory;
};

EquilibriumRunOptions parse_run_options(int argc, char** argv) {
    EquilibriumRunOptions options;
    auto require_value = [&](int& argument, const char* name) -> std::string {
        if (++argument >= argc) {
            throw std::runtime_error(std::string("Missing value for ") + name);
        }
        return argv[argument];
    };
    for (int argument = 1; argument < argc; ++argument) {
        const std::string name = argv[argument];
        if (name == "--steps") {
            options.steps = std::stoi(require_value(argument, "--steps"));
        } else if (name == "--output-every") {
            options.output_every =
                std::stoi(require_value(argument, "--output-every"));
        } else if (name == "--dt") {
            options.dt = std::stod(require_value(argument, "--dt"));
        } else if (name == "--markers-per-cell") {
            options.markers_per_occupied_cell =
                std::stoi(require_value(argument, "--markers-per-cell"));
        } else if (name == "--marker-budget") {
            options.marker_budget = static_cast<std::size_t>(
                std::stoull(require_value(argument, "--marker-budget")));
        } else if (name == "--radial-cells") {
            options.radial_cells =
                std::stoi(require_value(argument, "--radial-cells"));
        } else if (name == "--field-radial-cells") {
            options.field_radial_cells =
                std::stoi(require_value(argument, "--field-radial-cells"));
        } else if (name == "--snapshot-directory") {
            options.snapshot_directory =
                require_value(argument, "--snapshot-directory");
        } else if (name == "--threads") {
            options.threads = static_cast<unsigned int>(
                std::stoul(require_value(argument, "--threads")));
        } else if (name == "--shape") {
            const std::string shape = require_value(argument, "--shape");
            if (shape == "cic") {
                options.shape = ev::WeightedPICParticleShape::cic;
            } else if (shape == "tsc") {
                options.shape = ev::WeightedPICParticleShape::tsc;
            } else {
                throw std::runtime_error("Weighted PIC shape must be cic or tsc.");
            }
        } else {
            throw std::runtime_error("Unknown equilibrium option: " + name);
        }
    }
    if (options.steps < 1 || options.output_every < 1 || !(options.dt > 0.0)
        || options.markers_per_occupied_cell < 10
        || options.markers_per_occupied_cell > 1000
        || options.marker_budget == 0 || options.radial_cells < 4
        || options.field_radial_cells < 4 || options.threads == 0) {
        throw std::runtime_error("Invalid weighted PIC equilibrium run options.");
    }
    return options;
}

void write_radial_snapshot(
    const std::filesystem::path& directory,
    int frame,
    double time,
    const ev::WeightedPICDepositedMoments& deposited,
    const ev::WeightedPICRadialMesh& mesh) {
    if (directory.empty()) {
        return;
    }
    std::filesystem::create_directories(directory);
    std::ostringstream name;
    name << "weighted_pic_plummer_" << std::setw(4) << std::setfill('0')
         << frame << ".csv";
    std::ofstream output(directory / name.str());
    if (!output) {
        throw std::runtime_error("Could not create weighted PIC radial snapshot.");
    }
    output << std::setprecision(17)
           << "time,radius,density,radial_mean,radial_variance,neff\n";
    for (std::size_t i = 0; i < deposited.density.size(); ++i) {
        const double density = deposited.density[i];
        const double mean = density > 0.0
            ? deposited.radial_momentum_density[i] / density
            : 0.0;
        const double raw_second = density > 0.0
            ? deposited.radial_second_moment_density[i] / density
            : 0.0;
        const double variance = std::max(0.0, raw_second - mean * mean);
        output << time << ',' << mesh.venus_center[i] << ',' << density << ','
               << mean << ',' << variance << ','
               << deposited.effective_marker_number[i] << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    const EquilibriumRunOptions run = parse_run_options(argc, argv);
    const ev::Params source_p = equilibrium_params(run.radial_cells);
    const ev::PhaseSpaceCellGeometry source_geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(source_p), source_p);
    const ev::WeightedPICFieldGrid field_grid =
        ev::make_weighted_pic_field_grid(source_p, run.field_radial_cells);
    const ev::Params& field_p = field_grid.parameters;
    const ev::WeightedPICRadialMesh& mesh = field_grid.mesh;
    const std::vector<double> cell_average =
        plummer_cell_averages(source_geometry, source_p);

    ev::WeightedPICInitializationOptions options;
    options.markers_per_occupied_cell = run.markers_per_occupied_cell;
    options.global_marker_budget = run.marker_budget;
    options.velocity_pairing = ev::WeightedPICVelocityPairing::required;
    options.analytic_rejection_max_attempts = 65536;
    ev::WeightedPICSamplerState sampler{UINT64_C(0x706c756d6d657231), 0};
    ev::WeightedPICInitializationResult initialized =
        ev::initialize_weighted_vlasov_pic_from_analytic_density(
            cell_average,
            source_geometry,
            source_p,
            options,
            sampler,
            plummer_df,
            plummer_cell_upper_bound);

    std::vector<ev::WeightedPICMarker> markers =
        std::move(initialized.markers);
    const double represented_mass = ev::weighted_pic_total_marker_mass(markers);
    const ev::WeightedPICDepositedMoments initial =
        ev::deposit_weighted_vlasov_pic(markers, mesh, run.shape);
    const ev::WeightedPICNoiseDiagnostics initial_noise =
        ev::diagnose_weighted_pic_noise(
            markers,
            mesh,
            UINT64_C(0xbb67ae8584caa73b),
            run.shape);
    const double initial_target_l1 = plummer_target_l1(initial, mesh);
    const double initial_noise_l1 =
        integrated_density_uncertainty(initial_noise, mesh, represented_mass);

    const ev::VlasovPoissonField initial_field =
        ev::solve_weighted_pic_poisson_field(initial, mesh, 0.0, field_p);
    const double initial_energy = discrete_total_energy(initial, initial_field);
    const double initial_core_half = enclosed_mesh_mass(initial, mesh, 0.5);
    const double initial_core_one = enclosed_mesh_mass(initial, mesh, 1.0);

    const ev::CharacteristicIntegratorPlan plan =
        ev::make_characteristic_integrator_plan(field_p);
    auto advance_markers = [&](
                               std::vector<ev::WeightedPICMarker>& particles,
                               double dt,
                               const ev::VlasovPoissonField& field) {
        if (run.threads == 1) {
            ev::advance_weighted_pic_markers_with_venus(
                particles,
                dt,
                field_grid.radial_coordinate,
                field_grid.radius,
                field,
                field_p,
                plan);
        } else {
            ev::advance_weighted_pic_markers_with_venus_parallel(
                particles,
                dt,
                field_grid.radial_coordinate,
                field_grid.radius,
                field,
                field_p,
                plan,
                run.threads);
        }
    };
    double time = 0.0;
    double first_noise_excess_time = -1.0;
    int snapshot_frame = 0;
    write_radial_snapshot(
        run.snapshot_directory, snapshot_frame++, 0.0, initial, mesh);
    std::cout << std::setprecision(10)
              << "# time shell_L1 target_L1 noise_envelope abs_current "
                 "relative_energy_drift core_r0.5_drift core_r1_drift\n"
              << "0 0 " << initial_target_l1 << ' ' << 2.0 * initial_noise_l1
              << " 0 0 0 0\n";
    const auto evolution_start = std::chrono::steady_clock::now();
    for (int step = 0; step < run.steps; ++step) {
        const ev::WeightedPICDepositedMoments old_deposit =
            ev::deposit_weighted_vlasov_pic(markers, mesh, run.shape);
        const ev::VlasovPoissonField old_field =
            ev::solve_weighted_pic_poisson_field(
                old_deposit, mesh, time, field_p);
        std::vector<ev::WeightedPICMarker> midpoint = markers;
        advance_markers(midpoint, 0.5 * run.dt, old_field);
        const ev::WeightedPICDepositedMoments midpoint_deposit =
            ev::deposit_weighted_vlasov_pic(midpoint, mesh, run.shape);
        const ev::VlasovPoissonField midpoint_field =
            ev::solve_weighted_pic_poisson_field(
                midpoint_deposit, mesh, time + 0.5 * run.dt, field_p);
        advance_markers(markers, run.dt, midpoint_field);
        time += run.dt;

        if ((step + 1) % run.output_every == 0 || step + 1 == run.steps) {
            const ev::WeightedPICDepositedMoments checkpoint =
                ev::deposit_weighted_vlasov_pic(markers, mesh, run.shape);
            const ev::WeightedPICNoiseDiagnostics checkpoint_noise =
                ev::diagnose_weighted_pic_noise(
                    markers,
                    mesh,
                    UINT64_C(0xbb67ae8584caa73b),
                    run.shape);
            const ev::VlasovPoissonField checkpoint_field =
                ev::solve_weighted_pic_poisson_field(
                    checkpoint, mesh, time, field_p);
            const double checkpoint_drift = shell_l1_difference(
                initial.shell_mass, checkpoint.shell_mass, represented_mass);
            const double checkpoint_noise_l1 = integrated_density_uncertainty(
                checkpoint_noise, mesh, represented_mass);
            const double checkpoint_noise_envelope =
                initial_noise_l1 + checkpoint_noise_l1;
            const double energy_drift = std::abs(
                discrete_total_energy(checkpoint, checkpoint_field)
                    - initial_energy)
                / std::max(std::abs(initial_energy), 1.0e-300);
            const double core_half_drift =
                (enclosed_mesh_mass(checkpoint, mesh, 0.5) - initial_core_half)
                / represented_mass;
            const double core_one_drift =
                (enclosed_mesh_mass(checkpoint, mesh, 1.0) - initial_core_one)
                / represented_mass;
            if (first_noise_excess_time < 0.0
                && checkpoint_drift > checkpoint_noise_envelope) {
                first_noise_excess_time = time;
            }
            std::cout << time << ' ' << checkpoint_drift << ' '
                      << plummer_target_l1(checkpoint, mesh) << ' '
                      << checkpoint_noise_envelope << ' '
                      << normalized_absolute_current(
                             checkpoint, represented_mass)
                      << ' ' << energy_drift << ' ' << core_half_drift << ' '
                      << core_one_drift << '\n';
            write_radial_snapshot(
                run.snapshot_directory,
                snapshot_frame++,
                time,
                checkpoint,
                mesh);
        }
    }

    const ev::WeightedPICDepositedMoments final =
        ev::deposit_weighted_vlasov_pic(markers, mesh, run.shape);
    const ev::WeightedPICNoiseDiagnostics final_noise =
        ev::diagnose_weighted_pic_noise(
            markers,
            mesh,
            UINT64_C(0xbb67ae8584caa73b),
            run.shape);
    const double final_noise_l1 =
        integrated_density_uncertainty(final_noise, mesh, represented_mass);
    const double shell_drift = shell_l1_difference(
        initial.shell_mass, final.shell_mass, represented_mass);
    const double current = normalized_absolute_current(final, represented_mass);
    const double mass_error = std::abs(
        final.deposited_mesh_mass - represented_mass);
    const double mass_tolerance = 128.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, represented_mass);
    const double noise_envelope = initial_noise_l1 + final_noise_l1;
    const double evolution_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - evolution_start).count();

    auto reference_sum = [](const auto& values, const auto& term) {
        double sum = 0.0;
        double correction = 0.0;
        for (const auto& value : values) {
            const double increment = term(value);
            const double updated = sum + increment;
            correction += std::abs(sum) >= std::abs(increment)
                ? (sum - updated) + increment
                : (increment - updated) + sum;
            sum = updated;
        }
        return sum + correction;
    };
    const double reference_markers = reference_sum(
        markers, [](const auto& marker) { return marker.mass; });
    const double reference_shells = reference_sum(
        final.shell_mass, [](double mass) { return mass; });
    std::cerr << std::setprecision(17)
              << "mass diagnostic: N=" << markers.size()
              << " marker_mass=" << markers.front().mass
              << " represented=" << represented_mass
              << " mesh=" << final.deposited_mesh_mass
              << " independent_markers=" << reference_markers
              << " independent_shells=" << reference_shells
              << " mass_error=" << mass_error
              << " tolerance=" << mass_tolerance
              << " fused_population_error=" << std::fma(
                     static_cast<double>(markers.size()), markers.front().mass,
                     -reference_markers)
              << '\n';

    assert(initialized.diagnostics.density_aware_in_cell_sampling);
    assert(initialized.diagnostics.velocity_paired);
    assert(initial_target_l1 < 0.12);
    assert(mass_error <= mass_tolerance);
    const bool default_smoke = run.steps == 100 && run.output_every == 100
        && run.dt == 0.01 && run.markers_per_occupied_cell == 24
        && run.marker_budget == 80000 && run.radial_cells == 24
        && run.field_radial_cells == 24 && run.threads == 1
        && run.shape == ev::WeightedPICParticleShape::cic;
    if (default_smoke) {
        assert(shell_drift < 0.08);
        assert(shell_drift < 4.0 * noise_envelope);
    }

    std::cout << std::setprecision(10)
              << "Plummer weighted-PIC equilibrium smoke\n"
              << "  markers=" << markers.size()
              << " occupied_source_cells="
              << initialized.diagnostics.occupied_source_cells << '\n'
              << "  represented_mass=" << represented_mass
              << " deposited_mass=" << final.deposited_mesh_mass
              << " mass_error=" << mass_error << '\n'
              << "  initial_target_shell_L1=" << initial_target_l1 << '\n'
              << "  t_final=" << time
              << " initial_final_shell_L1=" << shell_drift
              << " two_batch_noise_envelope=" << noise_envelope << '\n'
              << "  final_absolute_radial_current_per_mass=" << current << '\n'
              << "  threads=" << run.threads
              << " shape="
              << (run.shape == ev::WeightedPICParticleShape::cic ? "cic" : "tsc")
              << " evolution_wall_seconds=" << evolution_seconds << '\n'
              << "  first_shell_drift_above_noise_time="
              << first_noise_excess_time << '\n'
              << "weighted Vlasov PIC Plummer equilibrium passed\n";
    return 0;
}
