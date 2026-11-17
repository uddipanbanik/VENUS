// Broad cell-average kinetic regression suite covering geometry, conservative
// SLDG, delayed carriers, Hamiltonian reconstruction/reference, and reversible
// finite-angle Boltzmann operators.

#include "ev/boltzmann_poisson.hpp"
#include "ev/canonical_reversible_collision.hpp"
#include "ev/conservative_sldg_transport.hpp"
#include "ev/config.hpp"
#include "ev/hamiltonian_sldg_transport.hpp"
#include "ev/initial_data.hpp"
#include "ev/output.hpp"
#include "ev/reversible_finite_angle_boltzmann.hpp"
#include "ev/sampled_reversible_boltzmann.hpp"
#include "ev/streamed_reversible_boltzmann.hpp"

#include <cassert>
#include <iomanip>
#include <iostream>

namespace {

ev::Params base_params() {
    ev::Params p;
    p.physics_model = "boltzmann_poisson";
    p.vp_universe = "static";
    p.vp_interpolation = "conservative_sldg";
    p.kinetic_grid = "cell_average_eta";
    p.vp_timestep_mode = "frank_dynamical";
    p.vp_orbit_regularization = "auto";
    p.vp_orbit_energy_tolerance = 1.0e-7;
    p.vp_orbit_lc_relative_tolerance = 1.0e-9;
    p.vp_orbit_lc_absolute_tolerance = 1.0e-11;
    p.vp_orbit_lc_max_steps = 20000;
    p.vp_orbit_lc_max_refinements = 2;
    p.inner_boundary = "reflecting";
    p.outer_boundary = "outflow_vacuum";
    p.momentum_boundary = "outflow_vacuum";
    p.radial_grid = "log";
    p.r_min = 0.08;
    p.r_max = 4.0;
    p.w_max = 3.0;
    p.L_max = 3.0;
    p.w_stretch = 0.0;
    p.Nr = 8;
    p.Nw = 8;
    p.NL = 4;
    p.integrator = "semi_lagrangian_kick_drift_kick";
    p.vp_sldg_subcells = 2;
    p.vp_sldg_geometry_tolerance = 1.0e-12;
    p.threads = 2;
    return p;
}

void test_collision_step_info_merge() {
    ev::BPCollisionStepInfo total;
    total.explicit_dt_limit = 0.4;
    total.collision_time_estimate = 3.0;
    total.minimum_mean_free_path = 2.0;
    total.sampled_min_samples_per_batch = 64;
    total.sampled_max_operator_reference_l1 = 0.1;
    total.sampled_max_shell_seconds = 0.2;
    ev::BPCollisionStepInfo part;
    part.substeps = 2;
    part.explicit_dt_limit = 0.25;
    part.collision_time_estimate = 4.0;
    part.sampled_min_samples_per_batch = 32;
    part.sampled_projection_cache_hits = 11;
    part.sampled_projection_cache_misses = 7;
    part.sampled_reused_operator_shells = 3;
    part.sampled_max_operator_reference_l1 = 0.15;
    part.sampled_context_cpu_seconds = 0.3;
    part.sampled_estimate_cpu_seconds = 0.4;
    part.sampled_materialize_cpu_seconds = 0.5;
    part.sampled_advance_cpu_seconds = 0.6;
    part.sampled_max_shell_seconds = 0.7;
    part.minimum_mean_free_path = 1.5;
    part.reversible_reaction_updates = 13;
    part.reversible_explicit_fast_path_fallbacks = 17;
    part.reversible_implicit_reaction_updates = 19;
    part.reversible_implicit_iterations = 23;
    ev::merge_bp_collision_step_info(total, part);
    assert(total.substeps == 2);
    assert(total.explicit_dt_limit == 0.25);
    assert(total.collision_time_estimate == 3.0);
    assert(total.sampled_min_samples_per_batch == 32);
    assert(total.sampled_projection_cache_hits == 11);
    assert(total.sampled_projection_cache_misses == 7);
    assert(total.sampled_reused_operator_shells == 3);
    assert(total.sampled_max_operator_reference_l1 == 0.15);
    assert(total.sampled_context_cpu_seconds == 0.3);
    assert(total.sampled_estimate_cpu_seconds == 0.4);
    assert(total.sampled_materialize_cpu_seconds == 0.5);
    assert(total.sampled_advance_cpu_seconds == 0.6);
    assert(total.sampled_max_shell_seconds == 0.7);
    assert(total.minimum_mean_free_path == 1.5);
    assert(total.reversible_reaction_updates == 13);
    assert(total.reversible_explicit_fast_path_fallbacks == 17);
    assert(total.reversible_implicit_reaction_updates == 19);
    assert(total.reversible_implicit_iterations == 23);
}

void test_mass_weighted_collision_clock() {
    const std::vector<double> mass{1.0e-9, 1.0, 2.0};
    const std::vector<double> timescale{1.0e-4, 2.0, 3.0};
    assert(ev::bp_sampled_mass_weighted_collision_timescale(
        mass, timescale, 1.0e-6) == 2.0);
    assert(ev::bp_sampled_mass_weighted_collision_timescale(
        mass, timescale, 0.0) == 1.0e-4);
    assert(std::isinf(ev::bp_sampled_mass_weighted_collision_timescale(
        {0.0}, {std::numeric_limits<double>::infinity()}, 1.0e-6)));
}

void test_canonical_row_roundoff_positivity_repair() {
    ev::CanonicalOrbitRow row;
    row.phase_mass = {1.0, -1.0e-14, 2.0};
    row.phase_energy_mass = {0.5, -2.0e-15, 3.0};
    row.phase_energy2_mass = {0.3, -1.0e-15, 5.0};
    const long double before = std::accumulate(
        row.phase_mass.begin(), row.phase_mass.end(), 0.0L);
    const double first_mean = row.phase_energy_mass[0] / row.phase_mass[0];
    const double last_mean = row.phase_energy_mass[2] / row.phase_mass[2];
    std::size_t repaired_cells = 0;
    const double repaired_mass = ev::canonical_repair_row_mass_positivity(
        row, 1.0e-12, repaired_cells);
    const long double after = std::accumulate(
        row.phase_mass.begin(), row.phase_mass.end(), 0.0L);
    assert(repaired_cells == 1);
    assert(std::abs(repaired_mass - 1.0e-14) < 1.0e-28);
    assert(std::abs(after - before) < 8.0e-16L);
    assert(row.phase_mass[1] == 0.0);
    assert(row.phase_energy_mass[1] == 0.0);
    assert(row.phase_energy2_mass[1] == 0.0);
    assert(std::abs(
        row.phase_energy_mass[0] / row.phase_mass[0] - first_mean)
        < 1.0e-15);
    assert(std::abs(
        row.phase_energy_mass[2] / row.phase_mass[2] - last_mean)
        < 1.0e-15);

    const std::vector<double> positivity_time{0.1, 0.2, 0.3};
    const std::vector<double> positivity_mass{1.0e-5, 0.2, 0.3};
    assert(ev::canonical_mass_budgeted_positivity_timescale(
        positivity_time, positivity_mass, 0.0) == 0.1);
    const double budgeted_time =
        ev::canonical_mass_budgeted_positivity_timescale(
            positivity_time, positivity_mass, 1.0e-4);
    assert(budgeted_time > 0.2 && budgeted_time < 0.201);
    const double broad_budget_time =
        ev::canonical_mass_budgeted_positivity_timescale(
            positivity_time, positivity_mass, 1.0);
    assert(broad_budget_time > 0.7 && broad_budget_time < 0.8);
}

void test_high_order_cell_quadrature() {
    for (int points = 5; points <= 12; ++points) {
        const auto& rule = ev::phase_space_gauss_legendre_rule(points);
        assert(rule.first.size() == static_cast<std::size_t>(points));
        assert(rule.second.size() == static_cast<std::size_t>(points));
        for (int degree = 0; degree < 2 * points; ++degree) {
            double represented = 0.0;
            for (int q = 0; q < points; ++q) {
                represented +=
                    rule.second[static_cast<std::size_t>(q)]
                    * std::pow(
                        rule.first[static_cast<std::size_t>(q)],
                        degree);
            }
            const double exact =
                (degree % 2 == 0)
                ? 2.0 / static_cast<double>(degree + 1)
                : 0.0;
            assert(std::abs(represented - exact) < 2.0e-13);
        }
    }
}

void test_composite_eta_axis() {
    ev::Params p = base_params();
    p.NL = 48;
    p.L_max = 2000.0;
    p.kinetic_eta_grid = "composite_power";
    p.kinetic_eta_power = 10.0;
    p.kinetic_eta_join = 5000.0;
    p.kinetic_eta_inner_cells = 32;
    const ev::PhaseSpaceAxis composite = ev::make_eta_axis(p);

    ev::Params inner = p;
    inner.NL = 32;
    inner.L_max = 100.0;
    inner.kinetic_eta_grid = "power";
    const ev::PhaseSpaceAxis reference = ev::make_eta_axis(inner);
    assert(composite.edge.size() == 49);
    assert(composite.edge.front() == 0.0);
    assert(composite.edge[32] == 5000.0);
    assert(composite.edge.back() == 2.0e6);
    for (std::size_t q = 0; q < reference.edge.size(); ++q) {
        assert(std::abs(composite.edge[q] - reference.edge[q])
            <= 8.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, reference.edge[q]));
    }
    for (std::size_t q = 1; q < composite.edge.size(); ++q) {
        assert(composite.edge[q] > composite.edge[q - 1]);
        assert(composite.width[q - 1]
            == composite.edge[q] - composite.edge[q - 1]);
    }
}

void test_parallel_exception_propagation() {
    bool caught = false;
    try {
        ev::parallel_for_dynamic_chunks(
            0,
            8,
            2,
            1,
            [](int lo, int, int) {
                if (lo == 3) {
                    throw std::runtime_error("expected worker failure");
                }
            });
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught);
}

void test_cell_average_configuration_contracts() {
    ev::Params p = base_params();
    p.physics_model = "vlasov_poisson";
    p.bp_collision_operator = "none";
    p.w_center = 0.25;
    bool rejected_shifted_velocity_grid = false;
    try {
        ev::validate_params(p);
    } catch (const std::runtime_error& error) {
        rejected_shifted_velocity_grid =
            std::string(error.what()).find("w_center=0") != std::string::npos;
    }
    assert(rejected_shifted_velocity_grid);

    ev::Params parsed = base_params();
    ev::apply_param(
        parsed,
        "bp_sampled_heat_floor_fraction",
        "0.2",
        "cell-average configuration test");
    assert(parsed.bp_sampled_heat_floor_fraction == 0.2);
    ev::apply_param(
        parsed,
        "bp_reversible_active_shell_floor_fraction",
        "1e-10",
        "cell-average configuration test");
    assert(parsed.bp_reversible_active_shell_floor_fraction == 1.0e-10);
    ev::apply_param(
        parsed,
        "bp_sampled_outgoing_angle_samples",
        "6",
        "cell-average configuration test");
    assert(parsed.bp_sampled_outgoing_angle_samples == 6);
    ev::apply_param(
        parsed,
        "bp_sampled_rhs_floor_fraction",
        "0.25",
        "cell-average configuration test");
    assert(parsed.bp_sampled_rhs_floor_fraction == 0.25);
    ev::apply_param(
        parsed,
        "bp_collision_locality_max_ratio",
        "0.75",
        "cell-average configuration test");
    assert(parsed.bp_collision_locality_max_ratio == 0.75);
    ev::apply_param(
        parsed,
        "bp_collision_fail_on_nonlocal_cells",
        "false",
        "cell-average configuration test");
    assert(!parsed.bp_collision_fail_on_nonlocal_cells);
    ev::apply_param(
        parsed,
        "bp_sampled_fixed_geometric_sequence",
        "false",
        "cell-average configuration test");
    assert(!parsed.bp_sampled_fixed_geometric_sequence);
    ev::apply_param(
        parsed,
        "bp_sampled_reuse_operator_graph",
        "false",
        "cell-average configuration test");
    assert(!parsed.bp_sampled_reuse_operator_graph);
    ev::apply_param(
        parsed,
        "bp_sampled_operator_refresh_l1_fraction",
        "0.125",
        "cell-average configuration test");
    assert(parsed.bp_sampled_operator_refresh_l1_fraction == 0.125);
    ev::apply_param(
        parsed,
        "vp_inner_carrier_max_refinement",
        "7",
        "cell-average configuration test");
    assert(parsed.vp_inner_carrier_max_refinement == 7);
    ev::apply_param(
        parsed,
        "vp_inner_carrier_return_time_bin_width",
        "2.5e-4",
        "cell-average configuration test");
    assert(parsed.vp_inner_carrier_return_time_bin_width == 2.5e-4);
    ev::apply_param(
        parsed,
        "vp_inner_carrier_compress_min_packets",
        "1234",
        "cell-average configuration test");
    assert(parsed.vp_inner_carrier_compress_min_packets == 1234);

    const std::filesystem::path sampled_config =
        std::filesystem::temp_directory_path()
        / "venus_streamed_config_parser_test.ini";
    {
        std::ofstream output(sampled_config);
        assert(output.good());
        output
            << "[collisions]\n"
            << "bp_sampled_fixed_geometric_sequence = true\n"
            << "bp_sampled_reuse_operator_graph = true\n"
            << "bp_sampled_operator_refresh_l1_fraction = 0.25\n";
    }
    ev::Params ini_parsed;
    ev::load_config_file(sampled_config, ini_parsed);
    assert(ini_parsed.bp_sampled_fixed_geometric_sequence);
    assert(ini_parsed.bp_sampled_reuse_operator_graph);
    assert(ini_parsed.bp_sampled_operator_refresh_l1_fraction == 0.25);
    std::vector<std::string> cli_storage{
        "venus_test",
        "--config",
        sampled_config.string(),
        "--bp-sampled-fixed-geometric-sequence",
        "false",
        "--bp-sampled-reuse-operator-graph",
        "false",
        "--bp-sampled-operator-refresh-l1-fraction",
        "0.375",
    };
    std::vector<char*> cli_arguments;
    cli_arguments.reserve(cli_storage.size());
    for (std::string& argument : cli_storage) {
        cli_arguments.push_back(argument.data());
    }
    const ev::Params cli_parsed = ev::parse_args(
        static_cast<int>(cli_arguments.size()), cli_arguments.data());
    assert(!cli_parsed.bp_sampled_fixed_geometric_sequence);
    assert(!cli_parsed.bp_sampled_reuse_operator_graph);
    assert(cli_parsed.bp_sampled_operator_refresh_l1_fraction == 0.375);
    std::filesystem::remove(sampled_config);

    ev::Params metadata_params = base_params();
    metadata_params.bp_collision_operator =
        "boltzmann_reversible_streamed";
    metadata_params.bp_sampled_fixed_geometric_sequence = false;
    metadata_params.bp_sampled_reuse_operator_graph = true;
    metadata_params.bp_sampled_operator_refresh_l1_fraction = 0.125;
    const ev::MpiContext mpi;
    const ev::LDecomposition decomp =
        ev::make_l_decomposition(metadata_params.NL, mpi);
    const std::filesystem::path metadata_path =
        std::filesystem::temp_directory_path()
        / "venus_streamed_collision_metadata_test.json";
    ev::write_metadata(
        metadata_path,
        metadata_params,
        metadata_params,
        decomp,
        mpi);
    std::ifstream metadata_input(metadata_path);
    const std::string metadata_text{
        std::istreambuf_iterator<char>(metadata_input),
        std::istreambuf_iterator<char>()};
    assert(metadata_text.find(
        "\"bp_collision_update\": "
        "\"operator_split_compact_streamed_sampled_reversible_"
        "finite_angle_boltzmann\"") != std::string::npos);
    assert(metadata_text.find(
        "\"bp_reversible_integrator\": "
        "\"symmetric_reaction_extent_heun_backward_euler\"")
        != std::string::npos);
    assert(metadata_text.find(
        "\"bp_sampled_fixed_geometric_sequence\": false")
        != std::string::npos);
    assert(metadata_text.find(
        "\"bp_sampled_reuse_operator_graph\": true")
        != std::string::npos);
    assert(metadata_text.find(
        "\"bp_sampled_operator_refresh_l1_fraction\": 0.125")
        != std::string::npos);
    std::filesystem::remove(metadata_path);

    metadata_params.bp_collision_operator = "boltzmann_exact";
    const std::filesystem::path exact_metadata_path =
        std::filesystem::temp_directory_path()
        / "venus_exact_collision_metadata_test.json";
    ev::write_metadata(
        exact_metadata_path,
        metadata_params,
        metadata_params,
        decomp,
        mpi);
    std::ifstream exact_metadata_input(exact_metadata_path);
    const std::string exact_metadata_text{
        std::istreambuf_iterator<char>(exact_metadata_input),
        std::istreambuf_iterator<char>()};
    assert(exact_metadata_text.find(
        "\"bp_collision_update\": "
        "\"operator_split_deterministic_all_pair_finite_angle_"
        "boltzmann_quadrature\"") != std::string::npos);
    assert(exact_metadata_text.find("dsmc_quadrature")
        == std::string::npos);
    std::filesystem::remove(exact_metadata_path);
}

void test_reversible_omission_budget_guard() {
    ev::Params p = base_params();
    p.Nr = 3;
    p.bp_boltzmann_remap_max_skipped_fraction = 1.0e-6;
    p.bp_reversible_active_shell_floor_fraction = 1.0e-12;
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    std::vector<ev::BPReversibleShellResult> shell(
        static_cast<std::size_t>(p.Nr));
    shell[0].represented_activity = 1.0;
    shell[1].represented_activity = 1.0;
    shell[2].omitted_projection_activity = 1.0e-100;
    shell[2].omitted_fraction = 1.0;
    ev::bp_reversible_validate_omission_budgets(
        p.Nr,
        geometry,
        p,
        [&](int i) -> const ev::BPReversibleShellResult& {
            return shell[static_cast<std::size_t>(i)];
        });

    shell[1].omitted_projection_activity = 2.0e-6;
    shell[1].omitted_fraction =
        shell[1].omitted_projection_activity
        / (shell[1].represented_activity
           + shell[1].omitted_projection_activity);
    bool rejected_active_shell = false;
    try {
        ev::bp_reversible_validate_omission_budgets(
            p.Nr,
            geometry,
            p,
            [&](int i) -> const ev::BPReversibleShellResult& {
                return shell[static_cast<std::size_t>(i)];
            });
    } catch (const std::runtime_error&) {
        rejected_active_shell = true;
    }
    assert(rejected_active_shell);
}

void test_phase_space_geometry_and_ledger() {
    ev::Params p = base_params();
    const ev::PhaseSpaceAxis radial = ev::make_positive_radial_cell_axis(p);
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(radial, p);
    assert(geometry.radial.edge.front() == p.r_min);
    assert(geometry.radial.edge.back() == p.r_max);
    assert(ev::phase_space_axis_is_uniform(geometry.radial_velocity.width));
    assert(!ev::phase_space_axis_is_uniform(geometry.eta.width));
    for (int k = 0; k < p.NL; ++k) {
        assert(std::abs(
            geometry.L[static_cast<std::size_t>(k)]
                * geometry.legacy_Lq[static_cast<std::size_t>(k)]
            - geometry.eta.width[static_cast<std::size_t>(k)]) < 1.0e-14);
    }
    std::vector<double> f(
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 1.0);
    const ev::KineticInvariantLedger ledger =
        ev::evaluate_kinetic_invariant_ledger(f, geometry, p);
    const double expected_mass =
        8.0 * ev::PI * ev::PI
        * (geometry.radial.edge.back() - geometry.radial.edge.front())
        * (geometry.radial_velocity.edge.back()
           - geometry.radial_velocity.edge.front())
        * (geometry.eta.edge.back() - geometry.eta.edge.front());
    assert(ev::kinetic_relative_error(ledger.mass, expected_mass) < 1.0e-13);
    assert(ledger.minimum_f == 1.0);

    ev::Params stretched_p = p;
    stretched_p.w_stretch = 1.5;
    const ev::PhaseSpaceCellGeometry stretched_geometry =
        ev::make_cell_average_eta_geometry(radial, stretched_p);
    assert(!ev::phase_space_axis_is_uniform(
        stretched_geometry.radial_velocity.width));
    for (int j = 0; j < stretched_p.Nw; ++j) {
        const int reverse_j = stretched_p.Nw - 1 - j;
        assert(std::abs(
            stretched_geometry.radial_velocity.center[static_cast<std::size_t>(j)]
            + stretched_geometry.radial_velocity.center[
                static_cast<std::size_t>(reverse_j)]) < 2.0e-15);
        assert(std::abs(
            stretched_geometry.radial_velocity.width[static_cast<std::size_t>(j)]
            - stretched_geometry.radial_velocity.width[
                static_cast<std::size_t>(reverse_j)]) < 2.0e-15);
    }
    const ev::KineticInvariantLedger stretched_ledger =
        ev::evaluate_kinetic_invariant_ledger(f, stretched_geometry, stretched_p);
    assert(ev::kinetic_relative_error(
        stretched_ledger.mass, expected_mass) < 1.0e-13);

    const ev::Moments moments = ev::compute_vp_moments(
        f,
        geometry.radial.center,
        geometry.radial_velocity.center,
        geometry.L,
        geometry.radial_velocity.width,
        geometry.legacy_Lq,
        p);
    const ev::VlasovPoissonField field =
        ev::solve_vlasov_poisson_field(
            moments, geometry.radial.center, 0.0, p);
    assert(ev::kinetic_relative_error(ledger.mass, field.matter_mass) < 1.0e-13);
    assert(ev::kinetic_relative_error(ledger.mass, field.outer_source_mass) < 1.0e-13);
    const std::vector<double> x_r =
        ev::radial_coordinate_grid(geometry.radial.center, p);
    const ev::VPSample outer = ev::sample_vp_field(
        p.r_max,
        x_r,
        geometry.radial.center,
        field,
        p);
    assert(ev::kinetic_relative_error(
        outer.phi,
        -p.vp_gravity * ledger.mass / p.r_max) < 2.0e-13);
    assert(ev::kinetic_relative_error(
        outer.phi_r,
        p.vp_gravity * ledger.mass / (p.r_max * p.r_max)) < 2.0e-13);
    for (int i = 1; i < p.Nr; ++i) {
        const double edge = geometry.radial.edge[static_cast<std::size_t>(i)];
        const double epsilon =
            1.0e-10 * geometry.radial.width[static_cast<std::size_t>(i)];
        const ev::VPSample left = ev::sample_vp_field(
            edge - epsilon,
            x_r,
            geometry.radial.center,
            field,
            p);
        const ev::VPSample right = ev::sample_vp_field(
            edge + epsilon,
            x_r,
            geometry.radial.center,
            field,
            p);
        assert(std::abs(left.phi - right.phi)
            < 1.0e-8 * std::max(1.0, std::abs(left.phi)));
        assert(std::abs(left.phi_r - right.phi_r)
            < 1.0e-8 * std::max(1.0, std::abs(left.phi_r)));
    }
}

void test_restart_representation_guard() {
    ev::Params p = base_params();
    const std::size_t count =
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL;
    std::vector<double> f(count, 0.0);
    for (std::size_t q = 0; q < count; ++q) {
        f[q] = 0.25 + static_cast<double>(q) / static_cast<double>(count);
    }
    std::vector<std::array<double, 5>> modes(count);
    for (std::size_t q = 0; q < count; ++q) {
        for (std::size_t mode = 0; mode < 5; ++mode) {
            modes[q][mode] = 1.0e-3
                * static_cast<double>((q + 1) * (mode + 1));
        }
    }
    const auto stamp =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::filesystem::path root =
        std::filesystem::temp_directory_path()
        / ("venus_fully_kinetic_restart_test_" + std::to_string(stamp));
    const std::filesystem::path restart = root / "restart_state.bin";
    const std::filesystem::path staging = root / "staging";
    ev::write_restart_state(
        restart, staging, p, 0.125, 7, f,
        nullptr, nullptr, nullptr, &modes);
    double time = 0.0;
    int step = 0;
    std::vector<std::array<double, 5>> loaded_modes;
    bool has_modes = false;
    const std::vector<double> loaded = ev::read_restart_state(
        restart,
        p,
        time,
        step,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &loaded_modes,
        &has_modes);
    assert(loaded == f);
    assert(has_modes);
    assert(loaded_modes == modes);
    assert(time == 0.125);
    assert(step == 7);

    ev::Params wrong = p;
    wrong.kinetic_grid = "legacy_nodal";
    bool rejected = false;
    try {
        (void)ev::read_restart_state(restart, wrong, time, step);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);

    ev::Params wrong_eta = p;
    wrong_eta.kinetic_eta_power = p.kinetic_eta_power + 0.5;
    rejected = false;
    try {
        (void)ev::read_restart_state(restart, wrong_eta, time, step);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);

    std::vector<double> invalid = f;
    invalid.front() = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try {
        ev::write_restart_state(
            root / "invalid_restart.bin", staging, p, 0.0, 0, invalid);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);

    // Nodal states obey the same finite, nonnegative physical-state
    // invariant as cell averages. Serialization rejects non-finite values.
    ev::Params legacy = p;
    legacy.kinetic_grid = "legacy_nodal";
    rejected = false;
    try {
        ev::write_restart_state(
            root / "invalid_legacy_restart.bin",
            staging,
            legacy,
            0.0,
            0,
            invalid);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);

    auto rewrite_valid = [&]() {
        ev::write_restart_state(
            restart, staging, p, 0.125, 7, f,
            nullptr, nullptr, nullptr, &modes);
    };
    auto overwrite = [&](std::streamoff offset, const auto& value) {
        std::fstream stream(
            restart,
            std::ios::binary | std::ios::in | std::ios::out);
        assert(stream.good());
        stream.seekp(offset);
        stream.write(
            reinterpret_cast<const char*>(&value),
            static_cast<std::streamsize>(sizeof(value)));
        stream.close();
        assert(stream.good());
    };
    auto rejected_read = [&]() {
        try {
            (void)ev::read_restart_state(restart, p, time, step);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };

    rewrite_valid();
    const std::array<char, 16> unterminated_magic{
        'E', 'V', 'L', 'O', 'G', 'R', 'S', 'T',
        '1', 'X', 'X', 'X', 'X', 'X', 'X', 'X'};
    overwrite(0, unterminated_magic);
    assert(rejected_read());

    // Version 9 with the default power eta grid stores r_min at byte 52 and
    // time at byte 120.  Infinite grid metadata used to satisfy inf <= inf in
    // the relative comparator; non-finite clocks also need an explicit guard.
    rewrite_valid();
    const double infinite_r_min =
        std::numeric_limits<double>::infinity();
    overwrite(52, infinite_r_min);
    assert(rejected_read());

    rewrite_valid();
    const double invalid_time =
        std::numeric_limits<double>::quiet_NaN();
    overwrite(120, invalid_time);
    assert(rejected_read());

    // A monotone-looking coordinate vector ending in +infinity is not a
    // usable phase-space grid and must be rejected by the binary-IC reader.
    const std::filesystem::path invalid_ic = root / "invalid_grid.bin";
    {
        std::ofstream output(invalid_ic, std::ios::binary);
        assert(output.good());
        const char magic[16] = {
            'E', 'V', 'I', 'C', 'R', 'W', 'L', '1', '\0'};
        output.write(magic, sizeof(magic));
        const std::array<std::int32_t, 4> header{1, 3, 3, 2};
        output.write(
            reinterpret_cast<const char*>(header.data()),
            static_cast<std::streamsize>(sizeof(header)));
        const std::array<double, 3> radius{
            1.0, 2.0, std::numeric_limits<double>::infinity()};
        const std::array<double, 3> velocity{-1.0, 0.0, 1.0};
        const std::array<double, 2> angular_momentum{0.0, 1.0};
        const std::array<double, 18> distribution{};
        for (const auto& values : {radius, velocity}) {
            output.write(
                reinterpret_cast<const char*>(values.data()),
                static_cast<std::streamsize>(sizeof(values)));
        }
        output.write(
            reinterpret_cast<const char*>(angular_momentum.data()),
            static_cast<std::streamsize>(sizeof(angular_momentum)));
        output.write(
            reinterpret_cast<const char*>(distribution.data()),
            static_cast<std::streamsize>(sizeof(distribution)));
        output.close();
        assert(output.good());
    }
    rejected = false;
    try {
        (void)ev::read_initial_data_rwl(invalid_ic);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
    std::filesystem::remove_all(root);
}

double default_analytic_df(
    double r,
    double w,
    double eta,
    const ev::Params& p) {
    const double b = p.beta;
    const double norm =
        p.amp * std::sqrt(b / (2.0 * ev::PI)) * b / (r * r);
    const double energy_shift = std::sqrt(
        1.0 + (w - p.w_in) * (w - p.w_in) + 2.0 * eta / (r * r));
    return norm * std::exp(-b * (energy_shift - 1.0));
}

void test_analytic_cell_average_initialization() {
    ev::Params p = base_params();
    p.Nr = 4;
    p.Nw = 4;
    p.NL = 3;
    p.beta = 2.3;
    p.amp = 0.07;
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    const std::vector<double> mu_seed(static_cast<std::size_t>(p.Nr), 0.0);
    const std::vector<double> x_r =
        ev::radial_coordinate_grid(geometry.radial.center, p);

    p.kinetic_ic_quadrature_points = 1;
    const std::vector<double> q1 =
        ev::initial_distribution_cell_average_eta(
            geometry, mu_seed, x_r, p);
    p.kinetic_ic_quadrature_points = 2;
    const std::vector<double> q2 =
        ev::initial_distribution_cell_average_eta(
            geometry, mu_seed, x_r, p);
    p.kinetic_ic_quadrature_points = 4;
    const std::vector<double> q4 =
        ev::initial_distribution_cell_average_eta(
            geometry, mu_seed, x_r, p);

    const int i = 0;
    const int j = 1;
    const int k = 0;
    constexpr int fine_points = 40;
    double reference = 0.0;
    for (int ir = 0; ir < fine_points; ++ir) {
        const double r = geometry.radial.edge[static_cast<std::size_t>(i)]
            + (static_cast<double>(ir) + 0.5) / fine_points
                * geometry.radial.width[static_cast<std::size_t>(i)];
        for (int iw = 0; iw < fine_points; ++iw) {
            const double w =
                geometry.radial_velocity.edge[static_cast<std::size_t>(j)]
                + (static_cast<double>(iw) + 0.5) / fine_points
                    * geometry.radial_velocity.width[static_cast<std::size_t>(j)];
            for (int ie = 0; ie < fine_points; ++ie) {
                const double eta =
                    geometry.eta.edge[static_cast<std::size_t>(k)]
                    + (static_cast<double>(ie) + 0.5) / fine_points
                        * geometry.eta.width[static_cast<std::size_t>(k)];
                reference += default_analytic_df(r, w, eta, p);
            }
        }
    }
    reference /= static_cast<double>(
        fine_points * fine_points * fine_points);
    const std::size_t q = ev::idx(i, j, k, p);
    const double error1 = std::abs(q1[q] - reference);
    const double error2 = std::abs(q2[q] - reference);
    const double error4 = std::abs(q4[q] - reference);
    std::cout << "Analytic IC cell-average errors: q1=" << error1
              << " q2=" << error2 << " q4=" << error4 << '\n';
    assert(error2 < error1);
    assert(error4 < error2);
}

void test_specular_wall_reconstruction() {
    ev::Params p = base_params();
    p.Nr = 3;
    p.Nw = 4;
    p.NL = 2;
    p.outer_boundary = "reflecting";
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    std::vector<double> f(
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 2.0);
    const int j = 1;
    const int mirrored_j = p.Nw - 1 - j;
    const int k = 0;
    f[ev::idx(0, mirrored_j, k, p)] = 1.8;
    f[ev::idx(1, j, k, p)] = 2.2;
    const std::array<double, 2> inner =
        ev::sldg_limited_slopes(f, 0, j, k, geometry, p);
    assert(inner[0] > 0.0);

    f.assign(f.size(), 2.0);
    f[ev::idx(p.Nr - 2, j, k, p)] = 1.8;
    f[ev::idx(p.Nr - 1, mirrored_j, k, p)] = 2.2;
    const std::array<double, 2> outer =
        ev::sldg_limited_slopes(f, p.Nr - 1, j, k, geometry, p);
    assert(outer[0] > 0.0);
}

void test_reflection_aware_sldg_refinement() {
    ev::Params p = base_params();
    p.Nr = 3;
    p.Nw = 5;
    p.NL = 2;
    p.r_min = 0.5;
    p.r_max = 2.0;
    p.w_max = 2.0;
    p.L_max = 2.0;
    p.kinetic_eta_power = 6.0;
    p.beta = 6.0;
    p.vp_sldg_subcells = 2;
    p.vp_sldg_fallback_mass_tolerance = 0.99;
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    const std::vector<double>& r = geometry.radial.center;
    const std::vector<double>& w = geometry.radial_velocity.center;
    const std::vector<double>& L = geometry.L;
    const std::vector<double> x_r = ev::radial_coordinate_grid(r, p);
    const std::vector<double> mu_seed(static_cast<std::size_t>(p.Nr), 0.0);
    const std::vector<double> initial =
        ev::initial_distribution_cell_average_eta(
            geometry, mu_seed, x_r, p);
    const ev::Moments moments = ev::compute_vp_moments(
        initial,
        r,
        w,
        L,
        geometry.radial_velocity.width,
        geometry.legacy_Lq,
        p);
    const ev::VlasovPoissonField field =
        ev::solve_vlasov_poisson_field(moments, r, 0.0, p);
    const auto transported =
        ev::advect_vlasov_poisson_conservative_sldg_frozen(
            initial, x_r, r, w, L, field, geometry, 1.0e-7, p);
    const double fallback_fraction =
        transported.second.source_mass > 0.0
        ? transported.second.quadrature_fallback_mass
            / transported.second.source_mass
        : 0.0;
    const ev::KineticInvariantLedger before =
        ev::evaluate_kinetic_invariant_ledger(initial, geometry, p);
    const ev::KineticInvariantLedger after =
        ev::evaluate_kinetic_invariant_ledger(
            transported.first, geometry, p);
    std::cout << "Reflection-aware SLDG: fallback_fraction="
              << fallback_fraction
              << " traced=" << transported.second.traced_rectangles
              << " rejected=" << transported.second.rejected_rectangles
              << " traces=" << transported.second.characteristic_traces
              << " reused=" << transported.second.reused_characteristic_points
              << " max_work=" << transported.second.maximum_source_work
              << " largest_fallback_mass="
              << transported.second.largest_fallback_mass
              << " largest_fallback_r="
              << transported.second.largest_fallback_r
              << " largest_fallback_w="
              << transported.second.largest_fallback_w
              << " largest_fallback_L="
              << transported.second.largest_fallback_L
              << " largest_fallback_refinement="
              << transported.second.largest_fallback_refinement
              << '\n';
    assert(fallback_fraction <= 1.0e-9);
    assert(std::abs(
        before.mass - after.mass - transported.second.escaped_mass)
        < 1.0e-11 * std::max(1.0, before.mass));
}

void test_directional_sldg_refinement_choice() {
    ev::SLDGTracePatch radial_curve;
    ev::SLDGTracePatch velocity_curve;
    ev::SLDGTracePatch bilinear_map;
    for (int ir = 0; ir < 3; ++ir) {
        const double x = static_cast<double>(ir - 1);
        for (int iw = 0; iw < 3; ++iw) {
            const double y = static_cast<double>(iw - 1);
            radial_curve.at(ir, iw) = {x + 0.2 * x * x, y};
            velocity_curve.at(ir, iw) = {x, y + 0.2 * y * y};
            bilinear_map.at(ir, iw) = {x, y + 0.1 * x * y};
        }
    }
    assert(ev::sldg_trace_patch_prefers_radial_split(
        radial_curve, 2.0, 2.0, false));
    assert(!ev::sldg_trace_patch_prefers_radial_split(
        velocity_curve, 2.0, 2.0, true));
    assert(ev::sldg_trace_patch_prefers_radial_split(
        bilinear_map, 2.0, 2.0, true));
    assert(!ev::sldg_trace_patch_prefers_radial_split(
        bilinear_map, 2.0, 2.0, false));

    const std::array<long long, 2> weighted =
        ev::sldg_allocate_binary_refinement_work(100, 9.0, 1.0);
    assert(weighted[0] + weighted[1] == 100);
    assert(weighted[0] > weighted[1]);
    assert(weighted[0] >= 1 && weighted[1] >= 1);
    const std::array<long long, 2> tied =
        ev::sldg_allocate_binary_refinement_work(17, 0.0, 0.0);
    assert(tied[0] + tied[1] == 17);
    assert(std::abs(tied[0] - tied[1]) <= 1);
}

void test_extended_polygon_partition() {
    const double r0 = 1.234567890123e-3;
    const double dr = 7.500000620552782e-11;
    const double w0 = -0.431234567890123;
    const double dw = 5.630441119999999e-6;
    const ev::SLDGPolygon polygon{
        {r0 + 0.03 * dr, w0 + 0.07 * dw},
        {r0 + 1.01 * dr, w0},
        {r0 + 0.96 * dr, w0 + 1.02 * dw},
        {r0, w0 + 0.91 * dw},
    };
    const std::vector<double> radial_edges{
        r0 - 0.2 * dr,
        r0 + 0.23 * dr,
        r0 + 0.58 * dr,
        r0 + 0.83 * dr,
        r0 + 1.2 * dr,
    };
    const std::vector<double> velocity_edges{
        w0 + 0.05 * dw,
        w0 + 0.31 * dw,
        w0 + 0.67 * dw,
        w0 + 0.95 * dw,
    };
    const long double domain_area =
        ev::sldg_clip_rectangle_area_extended(
            polygon,
            radial_edges.front(),
            radial_edges.back(),
            velocity_edges.front(),
            velocity_edges.back());
    const ev::SLDGExtendedClipContext context(polygon);
    const ev::SLDGExtendedPolygonMoments domain_moments =
        context.rectangle_moments(
            radial_edges.front(),
            radial_edges.back(),
            velocity_edges.front(),
            velocity_edges.back());
    long double tiled_area = 0.0L;
    long double tiled_rr = 0.0L;
    long double tiled_rw = 0.0L;
    long double tiled_ww = 0.0L;
    for (std::size_t i = 0; i + 1 < radial_edges.size(); ++i) {
        for (std::size_t j = 0; j + 1 < velocity_edges.size(); ++j) {
            const ev::SLDGExtendedPolygonMoments tile =
                context.rectangle_moments(
                radial_edges[i],
                radial_edges[i + 1],
                velocity_edges[j],
                velocity_edges[j + 1]);
            tiled_area += tile.area;
            tiled_rr += tile.second_rr;
            tiled_rw += tile.second_rw;
            tiled_ww += tile.second_ww;
        }
    }
    const long double relative_defect =
        std::abs(tiled_area - domain_area) / domain_area;
    std::cout << "Extended polygon partition: area="
              << static_cast<double>(domain_area)
              << " relative_defect=" << static_cast<double>(relative_defect)
              << '\n';
    assert(domain_area > 0.0L);
    auto relative_moment_defect = [](long double sum, long double exact) {
        return std::abs(sum - exact)
            / std::max(std::abs(exact), std::numeric_limits<long double>::min());
    };
    std::cout << std::setprecision(std::numeric_limits<long double>::max_digits10)
              << "Polygon precision: long_double_digits="
              << std::numeric_limits<long double>::digits
              << " epsilon=" << std::numeric_limits<long double>::epsilon()
              << " area_relative=" << relative_defect
              << " rr_relative="
              << relative_moment_defect(tiled_rr, domain_moments.second_rr)
              << " rw_relative="
              << relative_moment_defect(tiled_rw, domain_moments.second_rw)
              << " ww_relative="
              << relative_moment_defect(tiled_ww, domain_moments.second_ww)
              << std::endl;
    std::cout << "Polygon whole moments: area=" << domain_moments.area
              << " rr=" << domain_moments.second_rr
              << " rw=" << domain_moments.second_rw
              << " ww=" << domain_moments.second_ww << std::endl;
    // The sum combines twelve clipped tiles and their physical-coordinate
    // moment transforms. The tolerance accounts for binary64 roundoff on
    // platforms whose long double has 53 mantissa bits, and uses 1e-16
    // on platforms with extended precision.
    const long double partition_tolerance = std::max(
        1.0e-16L, 16.0L * std::numeric_limits<long double>::epsilon());
    assert(relative_defect < partition_tolerance);
    assert(relative_moment_defect(tiled_rr, domain_moments.second_rr) < partition_tolerance);
    assert(relative_moment_defect(tiled_rw, domain_moments.second_rw) < partition_tolerance);
    assert(relative_moment_defect(tiled_ww, domain_moments.second_ww) < partition_tolerance);

    ev::SLDGSourcePolynomial polynomial;
    const double polynomial_r1 = r0 + dr;
    const double polynomial_w1 = w0 + dw;
    const double represented_dr = polynomial_r1 - r0;
    const double represented_dw = polynomial_w1 - w0;
    polynomial.cell_average = 1.0;
    polynomial.center_r = 0.5 * (r0 + polynomial_r1);
    polynomial.center_w = 0.5 * (w0 + polynomial_w1);
    polynomial.scale_r = represented_dr;
    polynomial.scale_w = represented_dw;
    polynomial.mode = {0.2, -0.1, -12.0, 0.3, 0.4};
    assert(polynomial.minimum_on_parent_cell() < 0.0);
    const double positivity_scale = polynomial.enforce_positivity();
    assert(positivity_scale > 0.0 && positivity_scale < 1.0);
    assert(polynomial.minimum_on_parent_cell() >= 0.0);

    ev::SLDGSourcePolynomial low_amplitude;
    low_amplitude.cell_average = 9.2594917329327991e-11;
    low_amplitude.mode = {{
        3.2656178676982881e-10,
        -3.2651244020563106e-10,
        1.6328088835867065e-09,
        -1.9590746110797416e-09,
        1.6320687690293261e-09,
    }};
    assert(low_amplitude.minimum_on_parent_cell() < 0.0);
    assert(low_amplitude.enforce_positivity() < 1.0);
    assert(low_amplitude.minimum_on_parent_cell() >= 0.0);
    assert(low_amplitude.rectangle_integral(-0.5, 0.0, -0.5, 0.0)
        >= 0.0);
    const double exact_parent_mass =
        polynomial.rectangle_integral(r0, polynomial_r1, w0, polynomial_w1);
    assert(std::abs(exact_parent_mass - represented_dr * represented_dw)
        < 2.0e-15 * represented_dr * represented_dw);

    const std::array<ev::SLDGAffineTriangle, 2> identity_triangle{{
        {{{{r0, w0}, {polynomial_r1, w0}, {polynomial_r1, polynomial_w1}}},
          {{{r0, w0}, {polynomial_r1, w0}, {polynomial_r1, polynomial_w1}}}},
        {{{{r0, w0}, {polynomial_r1, polynomial_w1}, {r0, polynomial_w1}}},
          {{{r0, w0}, {polynomial_r1, polynomial_w1}, {r0, polynomial_w1}}}},
    }};
    long double integrated_parent_mass = 0.0L;
    for (const ev::SLDGAffineTriangle& triangle : identity_triangle) {
        ev::SLDGExtendedPolygon extended;
        for (const ev::SLDGPoint& point : triangle.arrival) {
            extended.push_back({point.r, point.w});
        }
        integrated_parent_mass +=
            ev::sldg_integrate_source_polynomial_over_arrival_overlap(
                polynomial,
                triangle,
                ev::sldg_extended_polygon_moments(extended));
    }
    const long double quadratic_integral_relative_error = std::abs(
        integrated_parent_mass
        - static_cast<long double>(exact_parent_mass))
        / exact_parent_mass;
    std::cout << "Quadratic polygon integral relative error="
              << static_cast<double>(quadratic_integral_relative_error)
              << '\n';
    assert(quadratic_integral_relative_error < 2.0e-15L);
}

ev::VlasovPoissonField represented_test_field(
    const ev::PhaseSpaceCellGeometry& geometry,
    const ev::Params& p) {
    ev::Moments moments;
    moments.rho.resize(static_cast<std::size_t>(p.Nr), 0.0);
    moments.p.resize(static_cast<std::size_t>(p.Nr), 0.0);
    moments.pt.resize(static_cast<std::size_t>(p.Nr), 0.0);
    moments.j.resize(static_cast<std::size_t>(p.Nr), 0.0);
    for (int i = 0; i < p.Nr; ++i) {
        const double r = geometry.radial.center[static_cast<std::size_t>(i)];
        const double line_density = 1.0 * (1.0 + 0.2 * i);
        moments.rho[static_cast<std::size_t>(i)] =
            line_density / (4.0 * ev::PI * r * r);
    }
    return ev::solve_vlasov_poisson_field(
        moments, geometry.radial.center, 0.0, p);
}

std::vector<double> represented_energy_equilibrium(
    const ev::PhaseSpaceCellGeometry& geometry,
    const ev::Params& p,
    const ev::VlasovPoissonField& field,
    double beta,
    double inner_guard = 0.25,
    double outer_guard = 4.5) {
    std::vector<double> f(
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 0.0);
    const std::vector<double> x_r =
        ev::radial_coordinate_grid(geometry.radial.center, p);
    const double phi_inner = ev::sample_vp_field(
        inner_guard,
        x_r,
        geometry.radial.center,
        field,
        p).phi;
    const double phi_outer = ev::sample_vp_field(
        outer_guard,
        x_r,
        geometry.radial.center,
        field,
        p).phi;
    constexpr double gauss = 0.577350269189625764509148780501957456;
    for (int i = 0; i < p.Nr; ++i) {
        const double rc = geometry.radial.center[static_cast<std::size_t>(i)];
        const double hr = 0.5 * geometry.radial.width[static_cast<std::size_t>(i)];
        for (int j = 0; j < p.Nw; ++j) {
            const double wc =
                geometry.radial_velocity.center[static_cast<std::size_t>(j)];
            const double hw =
                0.5 * geometry.radial_velocity.width[static_cast<std::size_t>(j)];
            for (int k = 0; k < p.NL; ++k) {
                const double etac =
                    geometry.eta.center[static_cast<std::size_t>(k)];
                const double heta =
                    0.5 * geometry.eta.width[static_cast<std::size_t>(k)];
                double average = 0.0;
                for (double xr : {-gauss, gauss}) {
                    const double radius = rc + hr * xr;
                    for (double xw : {-gauss, gauss}) {
                        const double wr = wc + hw * xw;
                        for (double xe : {-gauss, gauss}) {
                            const double eta = etac + heta * xe;
                            const ev::VPSample sample = ev::sample_vp_field(
                                radius,
                                x_r,
                                geometry.radial.center,
                                field,
                                p);
                            const double energy =
                                0.5 * wr * wr + eta / (radius * radius)
                                + sample.phi;
                            const double orbit_cutoff = std::min(
                                phi_inner + eta / (inner_guard * inner_guard),
                                phi_outer + eta / (outer_guard * outer_guard));
                            const double binding = orbit_cutoff - energy;
                            if (radius > inner_guard
                                && radius < outer_guard
                                && binding > 0.0) {
                                average += 0.125
                                    * binding * binding * std::exp(-beta * energy);
                            }
                        }
                    }
                }
                f[ev::idx(i, j, k, p)] = average;
            }
        }
    }
    return f;
}

double weighted_l1_difference(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const ev::PhaseSpaceCellGeometry& geometry,
    const ev::Params& p) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t q = ev::idx(i, j, k, p);
                const double measure = ev::canonical_cell_measure(geometry, i, j, k);
                numerator += measure * std::abs(a[q] - b[q]);
                denominator += measure * std::abs(a[q]);
            }
        }
    }
    return numerator / denominator;
}

double harmonic_transport_error(ev::Params p, double dt) {
    const ev::PhaseSpaceAxis radial =
        ev::make_positive_radial_cell_axis(p);
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(radial, p);
    const std::vector<double>& r = geometry.radial.center;
    const std::vector<double>& w = geometry.radial_velocity.center;
    const std::vector<double>& L = geometry.L;
    const std::vector<double> x_r = ev::radial_coordinate_grid(r, p);
    const ev::VlasovPoissonField field = represented_test_field(geometry, p);
    const std::vector<double> initial =
        represented_energy_equilibrium(geometry, p, field, 1.2);
    const auto transported =
        ev::advect_vlasov_poisson_conservative_sldg_frozen(
            initial, x_r, r, w, L, field, geometry, dt, p);
    return weighted_l1_difference(initial, transported.first, geometry, p);
}

struct ShellLedger {
    double mass = 0.0;
    double momentum = 0.0;
    double energy = 0.0;
    double H = 0.0;
};

ShellLedger shell_ledger(
    const std::vector<double>& shell_f,
    double radius,
    const ev::PhaseSpaceCellGeometry& geometry,
    const ev::Params& p) {
    ShellLedger ledger;
    for (int q = 0; q < p.Nw * p.NL; ++q) {
        const std::array<int, 2> indices =
            ev::bp_reversible_velocity_indices(q, p);
        const double measure =
            2.0 * ev::PI
            * geometry.radial_velocity.width[static_cast<std::size_t>(indices[0])]
            * geometry.eta.width[static_cast<std::size_t>(indices[1])]
            / (radius * radius);
        const double value = shell_f[static_cast<std::size_t>(q)];
        const double wr = ev::bp_reversible_radial_velocity(
            q, geometry.radial_velocity.center, p);
        const double energy = ev::bp_reversible_energy(
            q,
            radius,
            geometry.radial_velocity.center,
            geometry.eta.center,
            p);
        ledger.mass += measure * value;
        ledger.momentum += measure * wr * value;
        ledger.energy += measure * energy * value;
        if (value > 0.0) {
            ledger.H += measure * value * (std::log(value) - 1.0);
        }
    }
    return ledger;
}

void test_conservative_sldg_identity_and_equilibrium() {
    ev::Params p = base_params();
    p.Nr = 6;
    p.Nw = 6;
    p.NL = 3;
    p.r_min = 0.08;
    p.r_max = 6.0;
    p.w_max = 4.0;
    p.L_max = 3.0;
    const ev::PhaseSpaceAxis radial = ev::make_positive_radial_cell_axis(p);
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(radial, p);
    const std::vector<double>& r = geometry.radial.center;
    const std::vector<double>& w = geometry.radial_velocity.center;
    const std::vector<double>& L = geometry.L;
    const std::vector<double> x_r = ev::radial_coordinate_grid(r, p);
    const ev::VlasovPoissonField field = represented_test_field(geometry, p);
    const std::vector<double> initial =
        represented_energy_equilibrium(geometry, p, field, 1.2);

    const auto identity = ev::advect_vlasov_poisson_conservative_sldg_frozen(
        initial, x_r, r, w, L, field, geometry, 0.0, p);
    const double identity_error =
        weighted_l1_difference(initial, identity.first, geometry, p);
    std::cout << "SLDG identity L1=" << identity_error << '\n';
    assert(identity_error < 1.0e-11);
    const ev::KineticInvariantLedger before =
        ev::evaluate_kinetic_invariant_ledger(initial, geometry, p);
    const ev::KineticInvariantLedger after_identity =
        ev::evaluate_kinetic_invariant_ledger(identity.first, geometry, p);
    assert(ev::kinetic_relative_error(before.mass, after_identity.mass) < 1.0e-13);

    std::vector<double> invalid = initial;
    invalid.front() = -1.0;
    bool rejected = false;
    try {
        (void)ev::advect_vlasov_poisson_conservative_sldg_frozen(
            invalid, x_r, r, w, L, field, geometry, 0.0, p);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
    std::vector<double> wrong_w = w;
    wrong_w.front() = std::nextafter(wrong_w.front(), 0.0);
    rejected = false;
    try {
        (void)ev::advect_vlasov_poisson_conservative_sldg_frozen(
            initial, x_r, r, wrong_w, L, field, geometry, 0.0, p);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);

    p.vp_sldg_subcells = 1;
    const auto coarse = ev::advect_vlasov_poisson_conservative_sldg_frozen(
        initial, x_r, r, w, L, field, geometry, 1.0e-5, p);
    p.vp_sldg_subcells = 2;
    const auto fine = ev::advect_vlasov_poisson_conservative_sldg_frozen(
        initial, x_r, r, w, L, field, geometry, 1.0e-5, p);
    ev::Params serial_transport = p;
    serial_transport.threads = 1;
    const auto fine_serial =
        ev::advect_vlasov_poisson_conservative_sldg_frozen(
            initial,
            x_r,
            r,
            w,
            L,
            field,
            geometry,
            1.0e-5,
            serial_transport);
    assert(weighted_l1_difference(
        fine_serial.first, fine.first, geometry, p) < 1.0e-14);
    p.vp_sldg_subcells = 3;
    const auto finer = ev::advect_vlasov_poisson_conservative_sldg_frozen(
        initial, x_r, r, w, L, field, geometry, 1.0e-5, p);
    const double coarse_error =
        weighted_l1_difference(initial, coarse.first, geometry, p);
    const double fine_error =
        weighted_l1_difference(initial, fine.first, geometry, p);
    const double finer_error =
        weighted_l1_difference(initial, finer.first, geometry, p);
    const ev::KineticInvariantLedger after =
        ev::evaluate_kinetic_invariant_ledger(finer.first, geometry, p);
    std::cout << "SLDG mass before=" << before.mass
              << " after=" << after.mass
              << " escaped=" << finer.second.escaped_mass << '\n';
    assert(after.minimum_f >= 0.0);
    const double ledger_loss = before.mass - after.mass;
    const double remap_loss = finer.second.escaped_mass;
    assert(std::abs(ledger_loss - remap_loss)
        < 1.0e-11 * std::max(1.0, before.mass));
    assert(fine_error <= 1.2 * coarse_error);
    assert(finer_error <= 1.1 * fine_error);
    ev::Params refined_p = p;
    refined_p.Nr = 12;
    refined_p.Nw = 12;
    refined_p.NL = 6;
    refined_p.vp_sldg_subcells = 2;
    const double refined_grid_error =
        harmonic_transport_error(refined_p, 1.0e-5);
    assert(refined_grid_error < fine_error);
    std::cout << "SLDG represented-equilibrium L1: sub1=" << coarse_error
              << " sub2=" << fine_error
              << " sub3=" << finer_error
              << " refined_grid=" << refined_grid_error
              << " area_defect=" << finer.second.maximum_area_defect
              << " map_defect=" << finer.second.maximum_map_defect
              << " rejected_area_defect="
              << finer.second.maximum_rejected_area_defect
              << " rejected_map_defect="
              << finer.second.maximum_rejected_map_defect
              << " traced=" << finer.second.traced_rectangles
              << " rejected=" << finer.second.rejected_rectangles
              << " fallback_fraction="
              << finer.second.quadrature_fallback_mass
                    / finer.second.source_mass
              << '\n';
}

void test_hamiltonian_sldg_reconstruction_and_field_timestep(
    int eta_quadrature_points = 2) {
    {
        ev::Params cadence = base_params();
        cadence.dt_max = 1.0;
        cadence.vp_sldg_initial_transport_dt = 1.0e-4;
        cadence.vp_sldg_fallback_mass_tolerance = 1.0e-9;
        cadence.vp_sldg_target_fallback_fraction = 0.25;
        cadence.vp_sldg_timestep_growth = 1.5;
        cadence.vp_sldg_timestep_shrink = 0.5;
        ev::SLDGStepDiagnostics easy;
        easy.source_mass = 2.0;
        assert(std::abs(
            ev::sldg_next_transport_dt_cap(1.0e-4, easy, cadence)
            - 1.5e-4) < 1.0e-18);
        ev::SLDGStepDiagnostics on_target = easy;
        on_target.quadrature_fallback_mass =
            on_target.source_mass
            * cadence.vp_sldg_target_fallback_fraction
            * cadence.vp_sldg_fallback_mass_tolerance;
        assert(std::abs(
            ev::sldg_next_transport_dt_cap(
                1.0e-4, on_target, cadence)
            - 1.0e-4) < 1.0e-18);
        ev::SLDGStepDiagnostics near_limit = easy;
        near_limit.quadrature_fallback_mass =
            near_limit.source_mass
            * cadence.vp_sldg_fallback_mass_tolerance;
        assert(std::abs(
            ev::sldg_next_transport_dt_cap(
                1.0e-4, near_limit, cadence)
            - 5.0e-5) < 1.0e-18);
    }
    const double smooth_positive_theta =
        ev::hamiltonian_sldg_limit_positive_linear_reconstruction(
            1.0, 0.2, -0.1, 1.0, 1.0);
    assert(smooth_positive_theta == 1.0);
    const double positivity_theta =
        ev::hamiltonian_sldg_limit_positive_linear_reconstruction(
            1.0, 2.0, 2.0, 1.0, 1.0);
    assert(positivity_theta > 0.0 && positivity_theta < 0.251);
    assert(1.0 - 4.0 * positivity_theta >= 0.0);
    ev::Params grid_p = base_params();
    grid_p.Nr = 8;
    grid_p.Nw = 8;
    grid_p.NL = 4;
    grid_p.r_min = 0.08;
    grid_p.r_max = 6.0;
    grid_p.w_max = 4.0;
    grid_p.L_max = 3.0;
    grid_p.vp_sldg_subcells = 2;
    grid_p.vp_sldg_eta_quadrature_points = eta_quadrature_points;
    const ev::PhaseSpaceAxis radial =
        ev::make_positive_radial_cell_axis(grid_p);
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(radial, grid_p);
    const std::vector<double>& r = geometry.radial.center;
    const std::vector<double>& w = geometry.radial_velocity.center;
    const std::vector<double>& L = geometry.L;
    const std::vector<double> x_r =
        ev::radial_coordinate_grid(r, grid_p);
    const ev::VlasovPoissonField field =
        represented_test_field(geometry, grid_p);
    const std::vector<double> equilibrium =
        represented_energy_equilibrium(
            geometry, grid_p, field, 1.2);

    const auto grid_aligned =
        ev::advect_vlasov_poisson_conservative_sldg_frozen(
            equilibrium,
            x_r,
            r,
            w,
            L,
            field,
            geometry,
            1.0e-5,
            grid_p);
    ev::Params hamiltonian_p = grid_p;
    hamiltonian_p.physics_model = "vlasov_poisson";
    hamiltonian_p.vp_interpolation = "hamiltonian_sldg";
    ev::validate_params(hamiltonian_p);

    std::vector<double> subnormal_tail = equilibrium;
    const int tail_i = hamiltonian_p.Nr - 2;
    const int tail_j = 1;
    const int tail_k = hamiltonian_p.NL - 1;
    const std::size_t tail_cell =
        ev::idx(tail_i, tail_j, tail_k, hamiltonian_p);
    subnormal_tail[tail_cell] =
        32.0 * std::numeric_limits<double>::denorm_min();
    const ev::HamiltonianSLDGReconstruction tail_reconstruction =
        ev::build_hamiltonian_sldg_reconstruction(
            subnormal_tail,
            x_r,
            r,
            field,
            geometry,
            hamiltonian_p);
    assert(tail_reconstruction.slope[tail_cell][0] == 0.0);
    assert(tail_reconstruction.slope[tail_cell][1] == 0.0);
    const std::array<double, 2> grid_tail_slope = ev::sldg_limited_slopes(
        subnormal_tail,
        tail_i,
        tail_j,
        tail_k,
        geometry,
        hamiltonian_p);
    assert(grid_tail_slope[0] == 0.0);
    assert(grid_tail_slope[1] == 0.0);

    const auto hamiltonian =
        ev::advect_vlasov_poisson_hamiltonian_sldg_frozen(
            equilibrium,
            x_r,
            r,
            w,
            L,
            field,
            geometry,
            1.0e-5,
            hamiltonian_p);
    const double grid_error = weighted_l1_difference(
        equilibrium, grid_aligned.first, geometry, grid_p);
    const double hamiltonian_error = weighted_l1_difference(
        equilibrium, hamiltonian.first, geometry, hamiltonian_p);
    std::cout
        << "Hamiltonian SLDG equilibrium: grid=" << grid_error
        << " aligned=" << hamiltonian_error
        << " limited="
        << hamiltonian.second.hamiltonian_limited_cells
        << " deficient="
        << hamiltonian.second.hamiltonian_rank_deficient_cells
        << " phase_fraction="
        << hamiltonian.second.hamiltonian_maximum_phase_gradient_fraction
        << '\n';
    assert(hamiltonian_error < grid_error);
    const ev::KineticInvariantLedger equilibrium_before =
        ev::evaluate_kinetic_invariant_ledger(
            equilibrium, geometry, hamiltonian_p);
    const ev::KineticInvariantLedger equilibrium_after =
        ev::evaluate_kinetic_invariant_ledger(
            hamiltonian.first, geometry, hamiltonian_p);
    assert(std::abs(
            equilibrium_before.mass
            - equilibrium_after.mass
            - hamiltonian.second.escaped_mass)
        < 1.0e-11 * std::max(1.0, equilibrium_before.mass));

    std::vector<double> repeated = equilibrium;
    std::vector<std::array<double, 5>> repeated_modes;
    double repeated_escape = 0.0;
    for (int step = 0; step < 3; ++step) {
        std::vector<std::array<double, 5>> next_modes;
        auto update =
            ev::advect_vlasov_poisson_hamiltonian_sldg_frozen(
                repeated,
                x_r,
                r,
                w,
                L,
                field,
                geometry,
                1.0e-5,
                hamiltonian_p,
                nullptr,
                repeated_modes.empty() ? nullptr : &repeated_modes,
                &next_modes);
        repeated_escape += update.second.escaped_mass;
        repeated = std::move(update.first);
        repeated_modes = std::move(next_modes);
        assert(repeated_modes.size() == repeated.size());
    }
    const double repeated_error = weighted_l1_difference(
        equilibrium, repeated, geometry, hamiltonian_p);
    const ev::KineticInvariantLedger repeated_after =
        ev::evaluate_kinetic_invariant_ledger(
            repeated, geometry, hamiltonian_p);
    const double repeated_mass_residual = std::abs(
        equilibrium_before.mass
        - repeated_after.mass
        - repeated_escape);
    std::cout
        << "Hamiltonian SLDG repeated equilibrium: L1="
        << repeated_error
        << " mass_residual=" << repeated_mass_residual
        << '\n';
    assert(repeated_error < 4.0e-5);
    assert(repeated_mass_residual
        < 1.0e-11 * std::max(1.0, equilibrium_before.mass));

    std::vector<std::array<double, 5>> single_macro_modes;
    const auto single_macro =
        ev::advect_vlasov_poisson_hamiltonian_sldg_frozen(
            equilibrium,
            x_r,
            r,
            w,
            L,
            field,
            geometry,
            3.0e-5,
            hamiltonian_p,
            nullptr,
            nullptr,
            &single_macro_modes);
    const double single_macro_error = weighted_l1_difference(
        equilibrium, single_macro.first, geometry, hamiltonian_p);
    const ev::KineticInvariantLedger single_macro_after =
        ev::evaluate_kinetic_invariant_ledger(
            single_macro.first, geometry, hamiltonian_p);
    const double single_macro_mass_residual = std::abs(
        equilibrium_before.mass
        - single_macro_after.mass
        - single_macro.second.escaped_mass);
    std::cout
        << "Hamiltonian SLDG remap-count comparison: single="
        << single_macro_error
        << " repeated=" << repeated_error
        << " ratio=" << repeated_error / single_macro_error
        << '\n';
    assert(single_macro_modes.size() == equilibrium.size());
    assert(single_macro_mass_residual
        < 1.0e-11 * std::max(1.0, equilibrium_before.mass));
    assert(std::isfinite(single_macro_error));

    std::vector<double> perturbed = equilibrium;
    for (int i = 0; i < hamiltonian_p.Nr; ++i) {
        for (int j = 0; j < hamiltonian_p.Nw; ++j) {
            for (int k = 0; k < hamiltonian_p.NL; ++k) {
                const std::size_t q = ev::idx(i, j, k, hamiltonian_p);
                perturbed[q] *=
                    1.0
                    + 0.15
                        * w[static_cast<std::size_t>(j)]
                        / hamiltonian_p.w_max;
            }
        }
    }
    const ev::HamiltonianSLDGReconstruction reconstruction =
        ev::build_hamiltonian_sldg_reconstruction(
            perturbed,
            x_r,
            r,
            field,
            geometry,
            hamiltonian_p);
    assert(reconstruction.reconstructed_cells > 0);
    assert(reconstruction.maximum_phase_gradient_fraction > 1.0e-4);
    const auto perturbed_identity =
        ev::advect_vlasov_poisson_hamiltonian_sldg_frozen(
            perturbed,
            x_r,
            r,
            w,
            L,
            field,
            geometry,
            0.0,
            hamiltonian_p);
    assert(weighted_l1_difference(
        perturbed,
        perturbed_identity.first,
        geometry,
        hamiltonian_p) < 1.0e-11);

    ev::Moments moments;
    moments.rho.resize(static_cast<std::size_t>(hamiltonian_p.Nr), 0.0);
    moments.p.resize(static_cast<std::size_t>(hamiltonian_p.Nr), 0.0);
    moments.pt.resize(static_cast<std::size_t>(hamiltonian_p.Nr), 0.0);
    moments.j.resize(static_cast<std::size_t>(hamiltonian_p.Nr), 0.0);
    for (int i = 0; i < hamiltonian_p.Nr; ++i) {
        const double radius = r[static_cast<std::size_t>(i)];
        moments.rho[static_cast<std::size_t>(i)] =
            std::exp(-radius) + 0.05;
        moments.j[static_cast<std::size_t>(i)] =
            0.01 * radius
            * moments.rho[static_cast<std::size_t>(i)];
    }
    const ev::VlasovPoissonField varying_field =
        ev::solve_vlasov_poisson_field(
            moments, r, 0.0, hamiltonian_p);
    const ev::VlasovPoissonField field_rate =
        ev::vp_potential_time_derivative(
            moments, r, 0.0, hamiltonian_p);
    constexpr double epsilon = 1.0e-4;
    ev::Moments advanced_moments = moments;
    for (int i = 0; i < hamiltonian_p.Nr; ++i) {
        advanced_moments.rho[static_cast<std::size_t>(i)] +=
            epsilon
            * field_rate.source_line_density[static_cast<std::size_t>(i)]
            / (
                4.0 * ev::PI
                * r[static_cast<std::size_t>(i)]
                * r[static_cast<std::size_t>(i)]);
    }
    const ev::VlasovPoissonField advanced_field =
        ev::solve_vlasov_poisson_field(
            advanced_moments, r, epsilon, hamiltonian_p);
    double maximum_field_rate_error = 0.0;
    for (int i = 0; i < hamiltonian_p.Nr; ++i) {
        const double finite_difference =
            (advanced_field.phi[static_cast<std::size_t>(i)]
             - varying_field.phi[static_cast<std::size_t>(i)])
            / epsilon;
        const double scale = std::max(
            1.0e-12,
            std::abs(field_rate.phi[static_cast<std::size_t>(i)]));
        maximum_field_rate_error = std::max(
            maximum_field_rate_error,
            std::abs(
                finite_difference
                - field_rate.phi[static_cast<std::size_t>(i)])
                / scale);
    }
    const ev::VPPotentialVariationLimit potential_limit =
        ev::vp_potential_variation_limit(
            moments,
            varying_field,
            r,
            0.0,
            hamiltonian_p);
    std::cout
        << "Potential variation: tau=" << potential_limit.timescale
        << " dt_limit=" << potential_limit.timestep
        << " discrete_rate_error=" << maximum_field_rate_error
        << '\n';
    assert(maximum_field_rate_error < 1.0e-9);
    assert(std::isfinite(potential_limit.timescale));
    assert(std::abs(
            potential_limit.timestep
            - hamiltonian_p.vp_potential_variation_eta
                * potential_limit.timescale)
        < 1.0e-14 * potential_limit.timescale);

    ev::Moments faster_moments = moments;
    for (double& current : faster_moments.j) {
        current *= 2.0;
    }
    const ev::VPPotentialVariationLimit faster_limit =
        ev::vp_potential_variation_limit(
            faster_moments,
            varying_field,
            r,
            0.0,
            hamiltonian_p);
    assert(std::abs(
            faster_limit.timestep - 0.5 * potential_limit.timestep)
        < 2.0e-13 * potential_limit.timestep);

    ev::Moments steady_moments = moments;
    std::fill(
        steady_moments.j.begin(), steady_moments.j.end(), 0.0);
    const ev::VPPotentialVariationLimit steady_limit =
        ev::vp_potential_variation_limit(
            steady_moments,
            varying_field,
            r,
            0.0,
            hamiltonian_p);
    assert(!std::isfinite(steady_limit.timestep));

    ev::Params grid_aligned_p = hamiltonian_p;
    grid_aligned_p.vp_interpolation = "raw";
    const ev::VPPotentialVariationLimit grid_aligned_limit =
        ev::vp_potential_variation_limit(
            moments,
            varying_field,
            r,
            0.0,
            grid_aligned_p);
    assert(std::isfinite(grid_aligned_limit.timestep));
    assert(std::abs(
            grid_aligned_limit.timestep
            - grid_aligned_p.vp_potential_variation_eta
                * grid_aligned_limit.timescale)
        < 1.0e-14 * grid_aligned_limit.timescale);

    ev::Params weighted_p = hamiltonian_p;
    weighted_p.vp_timestep_mode = "active_cell_cfl";
    weighted_p.vp_timestep_omitted_mass_fraction = 1.0e-4;
    std::vector<double> sparse(equilibrium.size(), 0.0);
    const int bulk_i = weighted_p.Nr / 2;
    const int bulk_j = weighted_p.Nw / 2;
    const int bulk_k = weighted_p.NL / 2;
    sparse[ev::idx(bulk_i, bulk_j, bulk_k, weighted_p)] = 1.0;
    const double bulk_measure = ev::canonical_cell_measure(
        geometry, bulk_i, bulk_j, bulk_k);
    const double tail_measure = ev::canonical_cell_measure(
        geometry, 0, weighted_p.Nw - 1, 0);
    sparse[ev::idx(0, weighted_p.Nw - 1, 0, weighted_p)] =
        1.0e-8 * bulk_measure / tail_measure;
    const double mass_weighted_dt = ev::choose_dt_vlasov_poisson(
        sparse,
        r,
        w,
        L,
        varying_field,
        ev::min_spacing(x_r),
        ev::min_spacing(w),
        weighted_p,
        ev::MpiContext{},
        &geometry);
    weighted_p.vp_timestep_omitted_mass_fraction = 0.0;
    const double all_positive_cells_dt = ev::choose_dt_vlasov_poisson(
        sparse,
        r,
        w,
        L,
        varying_field,
        ev::min_spacing(x_r),
        ev::min_spacing(w),
        weighted_p,
        ev::MpiContext{},
        &geometry);
    assert(mass_weighted_dt > all_positive_cells_dt);
}

void test_hamiltonian_eta_quadrature_convergence() {
    ev::Params p = base_params();
    p.Nr = 8;
    p.Nw = 8;
    p.NL = 4;
    p.r_min = 0.08;
    p.r_max = 6.0;
    p.w_max = 4.0;
    p.L_max = 3.0;
    p.vp_sldg_subcells = 1;
    p.physics_model = "vlasov_poisson";
    p.vp_interpolation = "hamiltonian_sldg";
    p.vp_hamiltonian_reference = "none";
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    const std::vector<double>& r = geometry.radial.center;
    const std::vector<double>& w = geometry.radial_velocity.center;
    const std::vector<double>& L = geometry.L;
    const std::vector<double> x_r = ev::radial_coordinate_grid(r, p);
    const ev::VlasovPoissonField field =
        represented_test_field(geometry, p);
    const std::vector<double> equilibrium =
        represented_energy_equilibrium(geometry, p, field, 1.2);

    auto transport = [&](int eta_points) {
        ev::Params run = p;
        run.vp_sldg_eta_quadrature_points = eta_points;
        ev::validate_params(run);
        return ev::advect_vlasov_poisson_hamiltonian_sldg_frozen(
            equilibrium,
            x_r,
            r,
            w,
            L,
            field,
            geometry,
            1.0e-4,
            run);
    };
    const auto q1 = transport(1);
    const auto q2 = transport(2);
    const auto q4 = transport(4);
    const double q1_q4 = weighted_l1_difference(
        q4.first, q1.first, geometry, p);
    const double q2_q4 = weighted_l1_difference(
        q4.first, q2.first, geometry, p);
    std::size_t worst_cell = 0;
    double worst_q1_q4 = 0.0;
    for (std::size_t cell = 0; cell < q4.first.size(); ++cell) {
        const double error = std::abs(q1.first[cell] - q4.first[cell]);
        if (error > worst_q1_q4) {
            worst_q1_q4 = error;
            worst_cell = cell;
        }
    }
    const double same_cell_q2_q4 =
        std::abs(q2.first[worst_cell] - q4.first[worst_cell]);
    const ev::KineticInvariantLedger before =
        ev::evaluate_kinetic_invariant_ledger(equilibrium, geometry, p);
    for (const auto* result : {&q1, &q2, &q4}) {
        const ev::KineticInvariantLedger after =
            ev::evaluate_kinetic_invariant_ledger(
                result->first, geometry, p);
        assert(std::abs(
                before.mass - after.mass - result->second.escaped_mass)
            < 2.0e-11 * std::max(1.0, before.mass));
    }
    std::cout << "Hamiltonian eta quadrature: q1_vs_q4=" << q1_q4
              << " q2_vs_q4=" << q2_q4
              << " worst_q1_vs_q4=" << worst_q1_q4
              << " same_cell_q2_vs_q4=" << same_cell_q2_q4
              << " q2_eta_limited="
              << q2.second.hamiltonian_eta_limited_cells
              << " q4_eta_limited="
              << q4.second.hamiltonian_eta_limited_cells
              << '\n';
    assert(q2_q4 < q1_q4);
    assert(same_cell_q2_q4 < worst_q1_q4);
}

void test_hamiltonian_quadratic_reconstruction() {
    ev::Params p = base_params();
    p.physics_model = "vlasov_poisson";
    p.vp_interpolation = "hamiltonian_sldg";
    p.Nr = 8;
    p.Nw = 8;
    p.NL = 4;
    p.r_min = 0.08;
    p.r_max = 6.0;
    p.w_max = 4.0;
    p.L_max = 3.0;
    ev::validate_params(p);
    const ev::PhaseSpaceAxis radial = ev::make_positive_radial_cell_axis(p);
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(radial, p);
    const std::vector<double>& r = geometry.radial.center;
    const std::vector<double> x_r = ev::radial_coordinate_grid(r, p);
    const ev::VlasovPoissonField field = represented_test_field(geometry, p);
    const std::vector<double> equilibrium =
        represented_energy_equilibrium(geometry, p, field, 1.2);
    const ev::HamiltonianSLDGReconstruction reconstruction =
        ev::build_hamiltonian_sldg_reconstruction(
            equilibrium, x_r, r, field, geometry, p);
    assert(reconstruction.polynomial.size() == equilibrium.size());
    assert(reconstruction.reconstructed_cells > 0);
    assert(reconstruction.rank_deficient_cells == 0);
    long long occupied = 0;
    long long curved = 0;
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            for (int k = 0; k < p.NL; ++k) {
                const std::size_t q = ev::idx(i, j, k, p);
                if (!(equilibrium[q] > 0.0)) {
                    continue;
                }
                ++occupied;
                const ev::SLDGSourcePolynomial& polynomial =
                    reconstruction.polynomial[q];
                const double r0 = geometry.radial.edge[
                    static_cast<std::size_t>(i)];
                const double r1 = geometry.radial.edge[
                    static_cast<std::size_t>(i + 1)];
                const double w0 = geometry.radial_velocity.edge[
                    static_cast<std::size_t>(j)];
                const double w1 = geometry.radial_velocity.edge[
                    static_cast<std::size_t>(j + 1)];
                const double expected = equilibrium[q] * (r1 - r0) * (w1 - w0);
                const double represented =
                    polynomial.rectangle_integral(r0, r1, w0, w1);
                assert(std::abs(represented - expected)
                    < 2.0e-14 * std::max(expected, std::numeric_limits<double>::min()));
                const double minimum = polynomial.minimum_on_parent_cell();
                assert(minimum >= -1.0e-13 * equilibrium[q]);
                if (std::abs(polynomial.mode[2])
                        + std::abs(polynomial.mode[3])
                        + std::abs(polynomial.mode[4])
                    > 1.0e-12 * equilibrium[q]) {
                    ++curved;
                }
            }
        }
    }
    assert(occupied == reconstruction.reconstructed_cells);
    assert(curved > occupied / 4);
    const ev::HamiltonianOrbitReference orbit_reference =
        ev::build_hamiltonian_orbit_reference(
            equilibrium,
            reconstruction.polynomial,
            x_r,
            r,
            field,
            geometry,
            p);
    assert(orbit_reference.active_energy_planes > 0);
    assert(orbit_reference.residual_l1_fraction < 1.0);
    for (std::size_t q = 0; q < equilibrium.size(); ++q) {
        assert(orbit_reference.residual_polynomial[q].minimum_on_parent_cell()
            >= 0.0);
        assert(std::abs(
                orbit_reference.cell_average[q]
                + orbit_reference.residual_cell_average[q]
                - equilibrium[q])
            < 8.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, equilibrium[q]));
        for (std::size_t mode = 0; mode < 5; ++mode) {
            assert(std::abs(
                    orbit_reference.polynomial[q].mode[mode]
                    + orbit_reference.residual_polynomial[q].mode[mode]
                    - reconstruction.polynomial[q].mode[mode])
                < 64.0 * std::numeric_limits<double>::epsilon()
                    * std::max({
                        1.0,
                        std::abs(orbit_reference.polynomial[q].mode[mode]),
                        std::abs(
                            orbit_reference.residual_polynomial[q].mode[mode]),
                        std::abs(reconstruction.polynomial[q].mode[mode]),
                    }));
        }
    }
    long long eta_limited_cells = 0;
    const std::vector<ev::SLDGSourcePolynomial> eta_node_total =
        ev::build_hamiltonian_sldg_eta_node_polynomials(
            reconstruction.polynomial,
            geometry,
            p,
            &eta_limited_cells);
    const ev::HamiltonianOrbitReference eta_reference =
        ev::build_hamiltonian_orbit_reference_eta_quadrature(
            equilibrium,
            reconstruction.polynomial,
            x_r,
            r,
            field,
            geometry,
            p,
            &eta_node_total);
    const auto& eta_rule = ev::phase_space_gauss_legendre_rule(
        p.vp_sldg_eta_quadrature_points);
    assert(eta_reference.eta_node_residual_polynomial.size()
        == eta_rule.first.size() * equilibrium.size());
    for (std::size_t cell = 0; cell < equilibrium.size(); ++cell) {
        double averaged_total = 0.0;
        std::array<double, 5> averaged_total_mode{};
        double averaged_residual = 0.0;
        std::array<double, 5> averaged_mode{};
        for (std::size_t q_eta = 0; q_eta < eta_rule.first.size(); ++q_eta) {
            const double weight = 0.5 * eta_rule.second[q_eta];
            const ev::SLDGSourcePolynomial& node =
                eta_reference.eta_node_residual_polynomial[
                    q_eta * equilibrium.size() + cell];
            const ev::SLDGSourcePolynomial& total_node =
                eta_node_total[q_eta * equilibrium.size() + cell];
            assert(node.minimum_on_parent_cell() >= 0.0);
            assert(total_node.minimum_on_parent_cell()
                >= -256.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, equilibrium[cell]));
            averaged_total += weight * total_node.cell_average;
            averaged_residual += weight * node.cell_average;
            for (std::size_t mode = 0; mode < 5; ++mode) {
                averaged_total_mode[mode] += weight * total_node.mode[mode];
                averaged_mode[mode] += weight * node.mode[mode];
            }
        }
        assert(std::abs(averaged_total - equilibrium[cell])
            < 64.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, equilibrium[cell]));
        assert(std::abs(
                averaged_residual
                - eta_reference.residual_cell_average[cell])
            < 64.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, std::abs(averaged_residual)));
        assert(std::abs(
                eta_reference.cell_average[cell]
                + eta_reference.residual_cell_average[cell]
                - equilibrium[cell])
            < 64.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, equilibrium[cell]));
        for (std::size_t mode = 0; mode < 5; ++mode) {
            assert(std::abs(
                    averaged_total_mode[mode]
                    - reconstruction.polynomial[cell].mode[mode])
                < 128.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, std::abs(averaged_total_mode[mode])));
            assert(std::abs(
                    averaged_mode[mode]
                    - eta_reference.residual_polynomial[cell].mode[mode])
                < 128.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, std::abs(averaged_mode[mode])));
            assert(std::abs(
                    eta_reference.polynomial[cell].mode[mode]
                    + eta_reference.residual_polynomial[cell].mode[mode]
                    - reconstruction.polynomial[cell].mode[mode])
                < 128.0 * std::numeric_limits<double>::epsilon()
                    * std::max(
                        1.0,
                        std::abs(reconstruction.polynomial[cell].mode[mode])));
        }
    }
    std::cout << "Hamiltonian P2 reconstruction: occupied=" << occupied
              << " curved=" << curved
              << " positivity_scaled=" << reconstruction.limited_cells
              << " orbit_reference_mass_fraction="
              << orbit_reference.reference_mass_fraction
              << " orbit_residual_l1_fraction="
              << orbit_reference.residual_l1_fraction
              << " eta_orbit_residual_l1_fraction="
              << eta_reference.residual_l1_fraction
              << " eta_min_positive_reference_scale="
              << eta_reference.minimum_positive_reference_scale
              << " eta_limited_cells=" << eta_limited_cells
              << '\n';
}

void test_reflecting_wall_orbit_regularization() {
    ev::Params p = base_params();
    p.physics_model = "vlasov_poisson";
    p.vp_interpolation = "hamiltonian_sldg";
    p.Nr = 32;
    p.Nw = 16;
    p.NL = 4;
    p.r_min = 1.0e-3;
    p.r_max = 10.0;
    p.w_max = 3.0;
    p.L_max = 0.1;
    p.vp_orbit_energy_tolerance = 1.0e-9;
    p.vp_orbit_lc_relative_tolerance = 1.0e-9;
    p.vp_orbit_lc_absolute_tolerance = 1.0e-12;
    p.vp_orbit_lc_max_steps = 20000;
    p.vp_orbit_lc_max_refinements = 2;
    ev::validate_params(p);

    const ev::PhaseSpaceAxis radial =
        ev::make_positive_radial_cell_axis(p);
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(radial, p);
    const std::vector<double>& r = geometry.radial.center;
    const std::vector<double> x_r = ev::radial_coordinate_grid(r, p);
    const ev::VlasovPoissonField field =
        represented_test_field(geometry, p);
    const ev::CharacteristicIntegratorPlan plan =
        ev::make_characteristic_integrator_plan(p);
    const ev::CharacteristicState initial{
        0.0012441517084263913,
        -1.107421875,
    };
    const double angular_momentum = 0.0012195373516547571;
    const double dt = 0.00024073653543403272;
    const ev::VPUnregularizedAdvance ordinary =
        ev::advance_vp_characteristic_unregularized_result(
            initial.r,
            initial.w,
            angular_momentum,
            x_r,
            r,
            field,
            dt,
            p,
            plan);
    const double ordinary_error = ev::vp_frozen_hamiltonian_error(
        initial,
        ordinary.state,
        angular_momentum,
        x_r,
        r,
        field,
        p);
    const ev::VPOrbitRegularizationDiagnostics before =
        ev::vp_orbit_regularization_diagnostics();
    const ev::CharacteristicState corrected =
        ev::advance_vp_characteristic(
            initial.r,
            initial.w,
            angular_momentum,
            x_r,
            r,
            field,
            dt,
            p,
            plan);
    const ev::VPOrbitRegularizationDiagnostics after =
        ev::vp_orbit_regularization_diagnostics();
    const double corrected_error = ev::vp_frozen_hamiltonian_error(
        initial,
        corrected,
        angular_momentum,
        x_r,
        r,
        field,
        p);
    const ev::CharacteristicState reversed =
        ev::advance_vp_characteristic(
            corrected.r,
            corrected.w,
            angular_momentum,
            x_r,
            r,
            field,
            -dt,
            p,
            plan);
    const double reversal_error = std::hypot(
        (reversed.r - initial.r) / initial.r,
        reversed.w - initial.w);
    std::cout
        << "Reflecting-wall orbit: ordinary_energy=" << ordinary_error
        << " corrected_energy=" << corrected_error
        << " reversal=" << reversal_error
        << " wall_attempts=" << after.wall_retries - before.wall_retries
        << " wall_accepted=" << after.wall_accepted - before.wall_accepted
        << '\n';
    // The ordinary reflected kick exceeds the requested orbit tolerance.
    // The accepted regularized retry must reduce the frozen-Hamiltonian
    // error by the factor tested below.
    assert(ordinary_error > p.vp_orbit_energy_tolerance);
    assert(corrected_error < 1.0e-3 * ordinary_error);
    assert(corrected.r >= p.r_min);
    assert(corrected_error <= p.vp_orbit_energy_tolerance);
    assert(reversal_error < 2.0e-7);
    assert(after.wall_retries > before.wall_retries);
    assert(after.wall_accepted > before.wall_accepted);
    assert(after.wall_unresolved == before.wall_unresolved);
}

void test_delayed_inner_carrier_orbits_and_restart() {
    ev::Params p = base_params();
    p.physics_model = "vlasov_poisson";
    p.inner_boundary = "delayed_return";
    p.Nr = 4;
    p.Nw = 8;
    p.NL = 2;
    p.r_min = 0.1;
    p.r_max = 1.0;
    p.w_max = 2.0;
    p.L_max = 0.2;
    p.vp_sldg_subcells = 1;
    p.vp_inner_carrier_quadrature_points = 4;
    p.vp_inner_carrier_return_time_bin_width = 0.01;
    p.vp_inner_carrier_compress_min_packets = 1;
    p.vp_inner_carrier_max_packets = 1000;
    p.threads = 1;
    ev::validate_params(p);

    const double radial_flight =
        ev::delayed_inner_uniform_cell_round_trip_time(
            2.0, 0.0, 0.0, p);
    assert(std::abs(radial_flight - p.r_min) < 1.0e-14);
    const double angular_momentum = 0.08;
    const double A = 4.0
        + angular_momentum * angular_momentum
            / (p.r_min * p.r_min);
    const double pericentre2 =
        angular_momentum * angular_momentum / A;
    const double expected_flight = 2.0
        * std::sqrt(p.r_min * p.r_min - pericentre2)
        / std::sqrt(A);
    const double angular_flight =
        ev::delayed_inner_uniform_cell_round_trip_time(
            2.0, angular_momentum, 0.0, p);
    assert(std::abs(angular_flight - expected_flight)
        < 1.0e-14);
    const double central_frequency = 2.0;
    const double self_gravitating_mass =
        central_frequency * central_frequency
        * p.r_min * p.r_min * p.r_min / p.vp_gravity;
    const double self_gravitating_flight =
        ev::delayed_inner_uniform_cell_round_trip_time(
            0.0, 0.0, self_gravitating_mass, p);
    assert(std::abs(
            self_gravitating_flight
            - ev::PI / central_frequency)
        < 2.0e-14);

    auto free_rhs = [](const ev::CharacteristicState& state) {
        return ev::CharacteristicDerivative{state.w, 0.0};
    };
    const ev::RegularizedOrbitResult wall =
        ev::advance_radial_characteristic_reflecting_wall(
            {0.75, -1.0},
            0.5,
            0.5,
            free_rhs,
            1.0e-11,
            1.0e-13,
            10000);
    assert(wall.converged);
    assert(wall.wall_crossings == 1);
    assert(std::abs(wall.first_wall_crossing_time - 0.25) < 2.0e-10);
    assert(std::abs(wall.first_wall_crossing_velocity + 1.0) < 2.0e-10);
    assert(std::abs(wall.state.r - 0.75) < 2.0e-10);
    assert(std::abs(wall.state.w - 1.0) < 2.0e-10);

    auto constant_inward_rhs = [](const ev::CharacteristicState& state) {
        return ev::CharacteristicDerivative{state.w, -1.0};
    };
    const ev::RegularizedOrbitResult released_wall =
        ev::advance_radial_characteristic_reflecting_wall(
            {0.5, 1.0},
            2.25,
            0.5,
            constant_inward_rhs,
            1.0e-11,
            1.0e-13,
            10000,
            true);
    assert(released_wall.converged);
    assert(released_wall.wall_crossings == 1);
    assert(std::abs(
            released_wall.first_wall_crossing_time - 2.0)
        < 2.0e-9);
    assert(std::abs(
            released_wall.first_wall_crossing_velocity + 1.0)
        < 2.0e-9);

    const ev::PhaseSpaceAxis radial =
        ev::make_positive_radial_cell_axis(p);
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(radial, p);
    const std::vector<double>& r = geometry.radial.center;
    const std::vector<double>& w = geometry.radial_velocity.center;
    const std::vector<double>& L = geometry.L;
    const std::vector<double> x_r = ev::radial_coordinate_grid(r, p);

    ev::DelayedInnerCarrier compression_input;
    const ev::DelayedInnerCarrierPacket imminent_packet{
        7.0e-9, 1, geometry.eta.center[1], 0.4, 0.005};
    compression_input.packets.push_back(imminent_packet);
    for (int sample = 0; sample < 40; ++sample) {
        compression_input.packets.push_back({
            1.0e-9 * static_cast<double>(sample + 1),
            0,
            geometry.eta.center[0],
            0.10 + 0.01 * static_cast<double>(sample % 7),
            0.021 + 1.0e-4 * static_cast<double>(sample % 5),
        });
    }
    const double compression_time = 0.125;
    const ev::DelayedInnerCarrierMomentLedger compression_before =
        ev::evaluate_delayed_inner_carrier_moment_ledger(
            compression_input, geometry, compression_time, p);
    const ev::DelayedInnerCarrierCompressionLedger compression =
        ev::compress_delayed_inner_carrier(
            compression_input, geometry, compression_time, p);
    assert(compression.applied);
    assert(compression_input.packets.size()
        < compression_before.packet_count);
    const auto moment_close = [](double before, double after) {
        return std::abs(after - before)
            <= 2.0e-12 * std::max(std::abs(before), 1.0e-300);
    };
    assert(moment_close(
        compression.before.physical_mass,
        compression.after.physical_mass));
    assert(moment_close(
        compression.before.boundary_radial_momentum,
        compression.after.boundary_radial_momentum));
    assert(moment_close(
        compression.before.boundary_kinetic_energy,
        compression.after.boundary_kinetic_energy));
    assert(moment_close(
        compression.before.absolute_return_time_moment,
        compression.after.absolute_return_time_moment));
    int imminent_matches = 0;
    for (const ev::DelayedInnerCarrierPacket& packet
         : compression_input.packets) {
        if (packet.reduced_mass == imminent_packet.reduced_mass
            && packet.eta_index == imminent_packet.eta_index
            && packet.boundary_speed == imminent_packet.boundary_speed
            && packet.remaining_time == imminent_packet.remaining_time) {
            ++imminent_matches;
        }
    }
    assert(imminent_matches == 1);
    const std::size_t once_compressed = compression_input.packets.size();
    const ev::DelayedInnerCarrierCompressionLedger recompression =
        ev::compress_delayed_inner_carrier(
            compression_input, geometry, compression_time, p);
    assert(recompression.applied);
    assert(compression_input.packets.size() <= once_compressed);
    assert(moment_close(
        recompression.before.physical_mass,
        recompression.after.physical_mass));
    assert(moment_close(
        recompression.before.boundary_radial_momentum,
        recompression.after.boundary_radial_momentum));
    assert(moment_close(
        recompression.before.boundary_kinetic_energy,
        recompression.after.boundary_kinetic_energy));
    assert(moment_close(
        recompression.before.absolute_return_time_moment,
        recompression.after.absolute_return_time_moment));

    ev::Moments vacuum;
    vacuum.rho.assign(static_cast<std::size_t>(p.Nr), 0.0);
    vacuum.p.assign(static_cast<std::size_t>(p.Nr), 0.0);
    vacuum.pt.assign(static_cast<std::size_t>(p.Nr), 0.0);
    vacuum.j.assign(static_cast<std::size_t>(p.Nr), 0.0);
    const ev::VlasovPoissonField vacuum_field =
        ev::solve_vlasov_poisson_field(vacuum, r, 0.0, p, 0.0);
    const ev::SLDGPoint hidden = ev::sldg_trace_delayed_inner_arrival(
        0.15, -1.0, 0.0, x_r, r, vacuum_field, 0.1, p);
    assert(hidden.hidden);
    assert(hidden.touched_inner);
    assert(std::abs(hidden.remaining_hidden_time - 0.15) < 5.0e-9);
    const ev::SLDGPoint returned = ev::sldg_trace_delayed_inner_arrival(
        0.15, -1.0, 0.0, x_r, r, vacuum_field, 0.3, p);
    assert(!returned.hidden);
    assert(returned.touched_inner);
    assert(std::abs(returned.r - 0.15) < 5.0e-8);
    assert(std::abs(returned.w - 1.0) < 5.0e-8);

    std::vector<double> entering_f(
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 0.0);
    entering_f[ev::idx(0, 1, 0, p)] = 1.0;
    const ev::KineticInvariantLedger entering_before =
        ev::evaluate_kinetic_invariant_ledger(
            entering_f, geometry, p);
    ev::DelayedInnerCarrier captured_carrier;
    const auto captured =
        ev::advect_vlasov_poisson_conservative_sldg_frozen(
            entering_f,
            x_r,
            r,
            w,
            L,
            vacuum_field,
            geometry,
            0.1,
            p,
            &captured_carrier);
    const ev::KineticInvariantLedger captured_visible =
        ev::evaluate_kinetic_invariant_ledger(
            captured.first, geometry, p);
    const double captured_hidden =
        ev::evaluate_delayed_inner_carrier_ledger(
            captured_carrier, geometry, p).physical_mass;
    assert(captured_hidden > 0.5 * entering_before.mass);
    assert(std::abs(
            entering_before.mass
            - captured_visible.mass
            - captured_hidden
            - captured.second.escaped_mass)
        < 1.0e-13 * std::max(1.0, entering_before.mass));
    assert(std::abs(captured.second.carrier_mass_residual)
        < 1.0e-13 * std::max(1.0, entering_before.mass));
    const ev::VlasovPoissonField captured_field =
        ev::solve_vlasov_poisson_field(
            vacuum, r, 0.0, p, captured_hidden);
    assert(std::abs(
            captured_field.source_mass_edge.front() - captured_hidden)
        < 1.0e-14 * std::max(1.0, captured_hidden));

    ev::DelayedInnerCarrier carrier;
    carrier.packets.push_back({
        1.0e-5, 0, geometry.eta.center[0], 1.0, 0.02});
    const double carrier_mass =
        ev::evaluate_delayed_inner_carrier_ledger(
            carrier, geometry, p).physical_mass;
    const ev::VlasovPoissonField carrier_field =
        ev::solve_vlasov_poisson_field(
            vacuum, r, 0.0, p, carrier_mass);
    assert(std::abs(carrier_field.matter_mass - carrier_mass)
        < 1.0e-15);
    assert(std::abs(
            carrier_field.source_mass_edge.front() - carrier_mass)
        < 1.0e-15);
    std::vector<double> f(
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL, 0.0);
    const auto held = ev::advect_vlasov_poisson_conservative_sldg_frozen(
        f,
        x_r,
        r,
        w,
        L,
        carrier_field,
        geometry,
        0.01,
        p,
        &carrier);
    assert(carrier.packets.size() == 1);
    assert(std::abs(carrier.packets.front().remaining_time - 0.01)
        < 1.0e-15);
    assert(std::abs(held.second.carrier_mass_residual) < 1.0e-14);
    const auto released = ev::advect_vlasov_poisson_conservative_sldg_frozen(
        held.first,
        x_r,
        r,
        w,
        L,
        carrier_field,
        geometry,
        0.02,
        p,
        &carrier);
    const ev::KineticInvariantLedger visible =
        ev::evaluate_kinetic_invariant_ledger(
            released.first, geometry, p);
    const double hidden_after =
        ev::evaluate_delayed_inner_carrier_ledger(
            carrier, geometry, p).physical_mass;
    assert(std::abs(
            visible.mass + hidden_after
            + released.second.escaped_mass - carrier_mass)
        < 1.0e-13 * std::max(1.0, carrier_mass));
    assert(std::abs(released.second.carrier_mass_residual) < 1.0e-13);

    const auto stamp =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::filesystem::path root =
        std::filesystem::temp_directory_path()
        / ("venus_delayed_carrier_restart_test_" + std::to_string(stamp));
    const std::filesystem::path restart = root / "restart_state.bin";
    const std::filesystem::path staging = root / "staging";
    ev::DelayedInnerCarrier saved_carrier;
    saved_carrier.packets.push_back({
        2.0e-5, 1, geometry.eta.center[1], 0.7, 0.031});
    ev::write_restart_state(
        restart,
        staging,
        p,
        0.125,
        9,
        f,
        nullptr,
        &saved_carrier);
    double restart_time = 0.0;
    int restart_step = 0;
    ev::DelayedInnerCarrier loaded_carrier;
    const std::vector<double> loaded = ev::read_restart_state(
        restart,
        p,
        restart_time,
        restart_step,
        nullptr,
        nullptr,
        &loaded_carrier);
    assert(loaded == f);
    assert(restart_time == 0.125);
    assert(restart_step == 9);
    assert(loaded_carrier.packets.size() == 1);
    assert(loaded_carrier.packets.front().reduced_mass == 2.0e-5);
    assert(loaded_carrier.packets.front().eta_index == 1);
    assert(loaded_carrier.packets.front().boundary_speed == 0.7);
    assert(loaded_carrier.packets.front().remaining_time == 0.031);
    std::filesystem::remove_all(root);

    std::cout
        << "Delayed-return carrier: captured_fraction="
        << captured_hidden / entering_before.mass
        << " packets=" << captured_carrier.packets.size()
        << " compressed_packets=" << once_compressed
        << " mass_residual=" << captured.second.carrier_mass_residual
        << '\n';
}

void test_reversible_projection_and_maxwellian() {
    ev::Params p = base_params();
    p.Nr = 3;
    p.Nw = 6;
    p.NL = 3;
    p.r_min = 0.4;
    p.r_max = 2.0;
    p.w_max = 3.0;
    p.L_max = 3.0;
    p.w_stretch = 2.5;
    p.bp_collision_operator = "boltzmann_reversible";
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 0.2;
    p.bp_boltzmann_pair_phi_points = 2;
    p.bp_boltzmann_scatter_mu_points = 2;
    p.bp_boltzmann_scatter_phi_points = 2;
    p.bp_reversible_candidate_radius = 2;
    p.bp_reversible_max_velocity_cells = p.Nw * p.NL;
    p.bp_reversible_projection_tolerance = 1.0e-10;
    p.bp_boltzmann_mass_tol = 0.0;
    p.bp_boltzmann_remap_max_skipped_fraction = 0.5;
    p.bp_collision_cfl = 0.1;
    p.bp_collision_max_substeps = 100;
    const ev::PhaseSpaceAxis radial = ev::make_positive_radial_cell_axis(p);
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(radial, p);
    const int radial_index = 1;
    const double r = geometry.radial.center[static_cast<std::size_t>(radial_index)];
    {
        const std::vector<ev::BPReversiblePairCandidate> hull_pairs{
            ev::BPReversiblePairCandidate{0, 0, 0.0, 0.0, 0.0},
            ev::BPReversiblePairCandidate{1, 1, 0.0, 1.0, 0.0},
            ev::BPReversiblePairCandidate{2, 2, 0.0, 0.0, 1.0},
            ev::BPReversiblePairCandidate{3, 3, 0.0, 0.4, 0.2},
        };
        std::vector<double> hull_alpha;
        assert(ev::bp_reversible_solve_projection_hull(
            hull_pairs,
            0.25,
            0.35,
            1.0,
            1.0,
            1.0e-12,
            hull_alpha));
        double hull_mass = 0.0;
        double hull_momentum = 0.0;
        double hull_energy = 0.0;
        int hull_support = 0;
        for (std::size_t q = 0; q < hull_alpha.size(); ++q) {
            if (hull_alpha[q] > 0.0) {
                ++hull_support;
            }
            hull_mass += hull_alpha[q];
            hull_momentum += hull_alpha[q] * hull_pairs[q].momentum;
            hull_energy += hull_alpha[q] * hull_pairs[q].energy;
        }
        assert(hull_support <= 3);
        assert(std::abs(hull_mass - 1.0) < 1.0e-12);
        assert(std::abs(hull_momentum - 0.25) < 1.0e-12);
        assert(std::abs(hull_energy - 0.35) < 1.0e-12);

        std::vector<ev::BPReversibleNodeCandidate> candidates;
        unsigned domain_mask = ev::BP_REVERSIBLE_DOMAIN_NONE;
        assert(!ev::bp_reversible_node_candidates(
            geometry.radial_velocity.edge.back() + 1.0,
            geometry.eta.center.front(),
            geometry.radial_velocity.center,
            geometry.eta.center,
            geometry.radial_velocity.edge,
            geometry.eta.edge,
            p,
            1,
            candidates,
            &domain_mask));
        assert(domain_mask
            == ev::BP_REVERSIBLE_DOMAIN_RADIAL_VELOCITY);
        assert(!ev::bp_reversible_node_candidates(
            geometry.radial_velocity.center.front(),
            geometry.eta.edge.back() + 1.0,
            geometry.radial_velocity.center,
            geometry.eta.center,
            geometry.radial_velocity.edge,
            geometry.eta.edge,
            p,
            1,
            candidates,
            &domain_mask));
        assert(domain_mask
            == ev::BP_REVERSIBLE_DOMAIN_ANGULAR_MOMENTUM);
    }
    std::vector<double> shell_f(static_cast<std::size_t>(p.Nw * p.NL), 0.0);
    const double bulk = 0.15;
    const double temperature = 0.8;
    for (int j = 0; j < p.Nw; ++j) {
        const double wr = geometry.radial_velocity.center[static_cast<std::size_t>(j)];
        for (int k = 0; k < p.NL; ++k) {
            const double energy =
                0.5 * (wr - bulk) * (wr - bulk)
                + geometry.eta.center[static_cast<std::size_t>(k)] / (r * r);
            shell_f[static_cast<std::size_t>(
                ev::bp_reversible_velocity_index(j, k, p))] =
                std::exp(-energy / temperature);
        }
    }
    const ev::BPReversibleGraph graph = ev::bp_reversible_build_shell_graph(
        shell_f,
        r,
        geometry.radial_velocity.center,
        geometry.eta.center,
        geometry,
        p);
    assert(!graph.reactions.empty());
    const ev::BPFAVec3 projected_first{0.73, 0.51, 0.12};
    const ev::BPFAVec3 projected_second{-0.42, 0.36, -0.18};
    ev::BPReversibleProjection local_projection;
    bool projection_outside_domain = false;
    assert(ev::bp_reversible_project_outgoing_pair(
        projected_first,
        projected_second,
        r,
        geometry.radial_velocity.center,
        geometry.eta.center,
        geometry,
        p,
        local_projection,
        &projection_outside_domain));
    assert(!projection_outside_domain);
    assert(!local_projection.products.empty());
    assert(local_projection.products.size() <= 64);
    double projected_mass = 0.0;
    double projected_momentum = 0.0;
    double projected_energy = 0.0;
    for (const ev::BPReversibleProductPair& product
         : local_projection.products) {
        projected_mass += product.weight;
        projected_momentum += product.weight * (
            ev::bp_reversible_radial_velocity(
                product.first, geometry.radial_velocity.center, p)
            + ev::bp_reversible_radial_velocity(
                product.second, geometry.radial_velocity.center, p));
        projected_energy += product.weight * (
            ev::bp_reversible_energy(
                product.first,
                r,
                geometry.radial_velocity.center,
                geometry.eta.center,
                p)
            + ev::bp_reversible_energy(
                product.second,
                r,
                geometry.radial_velocity.center,
                geometry.eta.center,
                p));
    }
    const double target_projected_momentum =
        projected_first.x + projected_second.x;
    const double target_projected_energy = 0.5 * (
        ev::bp_fa_dot(projected_first, projected_first)
        + ev::bp_fa_dot(projected_second, projected_second));
    assert(std::abs(projected_mass - 1.0) < 1.0e-10);
    assert(std::abs(projected_momentum - target_projected_momentum)
        < 1.0e-10);
    assert(std::abs(projected_energy - target_projected_energy)
        < 1.0e-10);
    std::vector<double> zero_shell(shell_f.size(), 0.0);
    const ev::BPReversibleGraph zero_state_graph =
        ev::bp_reversible_build_shell_graph(
            zero_shell,
            r,
            geometry.radial_velocity.center,
            geometry.eta.center,
            geometry,
            p);
    assert(zero_state_graph.reactions.size() == graph.reactions.size());
    assert(zero_state_graph.omitted_domain_channels.size()
        == graph.omitted_domain_channels.size());
    assert(zero_state_graph.omitted_projection_channels.size()
        == graph.omitted_projection_channels.size());
    const std::vector<double> maxwellian_rhs =
        ev::bp_reversible_shell_rhs(shell_f, graph, p);
    double max_residual = 0.0;
    for (double value : maxwellian_rhs) {
        max_residual = std::max(max_residual, std::abs(value));
    }
    assert(max_residual < 2.0e-10);

    std::vector<double> perturbed = shell_f;
    perturbed[static_cast<std::size_t>(
        ev::bp_reversible_velocity_index(p.Nw / 2, p.NL / 2, p))] *= 1.25;
    const std::vector<double> rhs =
        ev::bp_reversible_shell_rhs(perturbed, graph, p);
    double mass_rhs = 0.0;
    double momentum_rhs = 0.0;
    double energy_rhs = 0.0;
    double entropy_rhs = 0.0;
    for (int q = 0; q < p.Nw * p.NL; ++q) {
        const std::array<int, 2> indices =
            ev::bp_reversible_velocity_indices(q, p);
        const double velocity_measure =
            2.0 * ev::PI
            * geometry.radial_velocity.width[static_cast<std::size_t>(indices[0])]
            * geometry.eta.width[static_cast<std::size_t>(indices[1])]
            / (r * r);
        const double wr = ev::bp_reversible_radial_velocity(
            q, geometry.radial_velocity.center, p);
        const double energy = ev::bp_reversible_energy(
            q, r, geometry.radial_velocity.center, geometry.eta.center, p);
        mass_rhs += velocity_measure * rhs[static_cast<std::size_t>(q)];
        momentum_rhs += velocity_measure * wr * rhs[static_cast<std::size_t>(q)];
        energy_rhs += velocity_measure * energy * rhs[static_cast<std::size_t>(q)];
        entropy_rhs += velocity_measure
            * std::log(perturbed[static_cast<std::size_t>(q)])
            * rhs[static_cast<std::size_t>(q)];
    }
    assert(std::abs(mass_rhs) < 1.0e-11);
    assert(std::abs(momentum_rhs) < 1.0e-10);
    assert(std::abs(energy_rhs) < 1.0e-10);
    assert(entropy_rhs <= 1.0e-11);

    ev::Params strict_p = p;
    strict_p.bp_boltzmann_remap_max_skipped_fraction = 1.0e-6;
    bool rejected_omission = false;
    try {
        const ev::BPReversibleShellResult strict_result =
            ev::bp_reversible_advance_shell(
            perturbed, graph, 1.0e-3, strict_p);
        ev::bp_reversible_validate_omission_budgets(
            1,
            geometry,
            strict_p,
            [&](int) -> const ev::BPReversibleShellResult& {
                return strict_result;
            });
    } catch (const std::runtime_error&) {
        rejected_omission = true;
    }
    assert(rejected_omission);

    const ShellLedger before = shell_ledger(perturbed, r, geometry, p);
    const ev::BPReversibleShellResult advanced =
        ev::bp_reversible_advance_shell(perturbed, graph, 1.0e-3, p);
    assert(ev::bp_reversible_nonnegative(advanced.f));
    const ShellLedger after = shell_ledger(advanced.f, r, geometry, p);
    assert(ev::kinetic_relative_error(before.mass, after.mass) < 2.0e-13);
    assert(std::abs(after.momentum - before.momentum)
        < 2.0e-13 * std::max(1.0, std::abs(before.momentum)));
    assert(ev::kinetic_relative_error(before.energy, after.energy) < 2.0e-13);
    assert(after.H <= before.H + 2.0e-13 * std::max(1.0, std::abs(before.H)));
    const double initial_explicit_limit =
        ev::bp_reversible_positive_dt_limit(
            perturbed, rhs, p.bp_collision_cfl);
    assert(initial_explicit_limit > 0.0);
    assert(std::isfinite(initial_explicit_limit));
    const ev::BPReversibleShellResult guarded_stiff_step =
        ev::bp_reversible_advance_shell(
            perturbed, graph, 2.0 * initial_explicit_limit, p);
    assert(std::isinf(guarded_stiff_step.minimum_dt_limit));
    const double comparison_dt = 1.0e-4;
    const ev::BPReversibleShellResult explicit_reference =
        ev::bp_reversible_advance_shell_explicit_ssprk2(
            perturbed, graph, comparison_dt, p);
    ev::Params implicit_p = p;
    implicit_p.bp_reversible_explicit_fast_path = false;
    const ev::BPReversibleShellResult implicit_full =
        ev::bp_reversible_advance_shell(
            perturbed, graph, comparison_dt, implicit_p);
    ev::BPReversibleShellResult implicit_half =
        ev::bp_reversible_advance_shell(
            perturbed, graph, 0.5 * comparison_dt, implicit_p);
    implicit_half = ev::bp_reversible_advance_shell(
        implicit_half.f, graph, 0.5 * comparison_dt, implicit_p);
    double full_error = 0.0;
    double half_error = 0.0;
    for (std::size_t q = 0; q < perturbed.size(); ++q) {
        const double volume = graph.velocity_cell_volume[q];
        full_error += volume * std::abs(
            implicit_full.f[q] - explicit_reference.f[q]);
        half_error += volume * std::abs(
            implicit_half.f[q] - explicit_reference.f[q]);
    }
    std::cout << "Implicit reaction-sweep errors: full="
              << full_error << " half=" << half_error << std::endl;
    assert(half_error <= 1.1 * full_error);
    std::cout << "Reversible Boltzmann: reactions=" << graph.reactions.size()
              << " omitted_fraction=" << advanced.omitted_fraction
              << " omitted_domain=" << graph.omitted_domain_activity
              << " omitted_projection=" << graph.omitted_projection_activity
              << " Maxwellian_max_residual=" << max_residual
              << " entropy_rhs=" << entropy_rhs
              << " implicit_errors=" << full_error << '/' << half_error
              << '\n';
}

void test_sampled_reversible_boltzmann() {
    ev::Params p = base_params();
    p.Nr = 3;
    p.Nw = 4;
    p.NL = 3;
    p.w_stretch = 2.5;
    p.r_min = 0.5;
    p.r_max = 2.0;
    p.w_max = 2.5;
    p.L_max = 2.5;
    p.bp_collision_operator = "boltzmann_reversible_sampled";
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 0.15;
    p.bp_boltzmann_pair_phi_points = 4;
    p.bp_boltzmann_scatter_mu_points = 2;
    p.bp_boltzmann_scatter_phi_points = 4;
    p.bp_reversible_candidate_radius = 2;
    p.bp_reversible_max_velocity_cells = p.Nw * p.NL;
    p.bp_reversible_projection_tolerance = 1.0e-10;
    p.bp_boltzmann_mass_tol = 0.0;
    p.bp_boltzmann_remap_max_skipped_fraction = 0.5;
    p.bp_sampled_dense_cells = 2;
    p.bp_sampled_energy_strata = 3;
    p.bp_sampled_L_strata = 2;
    p.bp_sampled_tail_fraction = 0.25;
    p.bp_sampled_stratified_fraction = 0.25;
    p.bp_sampled_uniform_fraction = 0.05;
    p.bp_sampled_tail_power = 1.5;
    p.bp_sampled_fail_on_unconverged = false;
    p.bp_sampled_ntc_pair_thinning = false;
    p.bp_sampled_seed = UINT64_C(1234567);
    ev::BPReversibleGraph duplicate_graph;
    ev::BPReversibleReaction duplicate_first;
    duplicate_first.incoming_first = 0;
    duplicate_first.incoming_second = 1;
    duplicate_first.coefficient = 2.0;
    duplicate_first.species = {
        ev::BPReversibleSpecies{0, 0.0, -1.0},
        ev::BPReversibleSpecies{1, 0.0, -1.0},
        ev::BPReversibleSpecies{2, 1.0, 1.0},
        ev::BPReversibleSpecies{3, 1.0, 1.0},
    };
    ev::BPReversibleReaction duplicate_second = duplicate_first;
    duplicate_second.coefficient = 3.0;
    duplicate_graph.reactions.push_back(duplicate_first);
    duplicate_graph.reactions.push_back(duplicate_second);
    ev::bp_sampled_coalesce_reaction_suffix(
        duplicate_graph, 0);
    assert(duplicate_graph.reactions.size() == 1);
    assert(duplicate_graph.reactions.front().coefficient == 5.0);
    assert(ev::bp_sampled_scatter_mu_from_unit(
        0.0, 1.0, p, "isotropic") == -1.0);
    assert(ev::bp_sampled_scatter_mu_from_unit(
        0.5, 1.0, p, "isotropic") == 0.0);
    assert(ev::bp_sampled_scatter_mu_from_unit(
        1.0, 1.0, p, "isotropic") == 1.0);
    for (const std::string& cross_section
         : {"constant", "powerlaw", "softened_gravity", "yukawa_born"}) {
        ev::Params bounded = p;
        bounded.bp_cross_section = cross_section;
        bounded.bp_sigma_u0 = 0.4;
        bounded.bp_sigma_n = 4.0;
        bounded.bp_subhalo_radius = 0.3;
        bounded.bp_subhalo_mass = 0.2;
        bounded.bp_coulomb_log = 2.0;
        bounded.bp_yukawa_alpha = 0.3;
        bounded.bp_yukawa_mchi = 2.0;
        bounded.bp_yukawa_mphi = 0.7;
        const double upper =
            ev::bp_sampled_u_sigma_upper_bound(4.0, bounded);
        assert(std::isfinite(upper) && upper > 0.0);
        for (int sample = 1; sample <= 4000; ++sample) {
            const double speed = 4.0 * sample / 4000.0;
            assert(
                speed * ev::bp_fa_total_cross_section(speed, bounded)
                <= upper * (1.0 + 2.0e-13));
        }
    }
    {
        constexpr int angular_samples = 8;
        std::array<int, 2> polar_counts{{0, 0}};
        std::array<int, 4> azimuth_counts{{0, 0, 0, 0}};
        for (int angle = 0; angle < angular_samples; ++angle) {
            const std::array<double, 2> unit =
                ev::bp_sampled_stratified_outgoing_units(
                    17, angle, angular_samples, 2, UINT64_C(12345));
            assert(unit[0] >= 0.0 && unit[0] < 1.0);
            assert(unit[1] >= 0.0 && unit[1] < 1.0);
            ++polar_counts[static_cast<std::size_t>(
                std::min(1, static_cast<int>(2.0 * unit[0])))];
            ++azimuth_counts[static_cast<std::size_t>(
                std::min(3, static_cast<int>(4.0 * unit[1])))];
        }
        assert(polar_counts[0] == 4 && polar_counts[1] == 4);
        for (int count : azimuth_counts) {
            assert(count == 2);
        }
        std::vector<double> mu;
        std::vector<double> mu_weight;
        std::vector<double> psi_cos;
        std::vector<double> psi_sin;
        ev::bp_fa_gauss_legendre_mu_table(2, mu, mu_weight);
        ev::bp_fa_midpoint_angle_table(4, psi_cos, psi_sin);
        std::array<int, 2> discrete_polar_counts{{0, 0}};
        std::array<int, 4> discrete_azimuth_counts{{0, 0, 0, 0}};
        for (int angle = 0; angle < angular_samples; ++angle) {
            const std::array<double, 2> unit =
                ev::bp_sampled_stratified_outgoing_units(
                    17, angle, angular_samples, 2, UINT64_C(12345));
            double sampled_mu = 0.0;
            double sampled_psi_cos = 0.0;
            double sampled_psi_sin = 0.0;
            ev::bp_sampled_discrete_scatter_direction(
                unit[0],
                unit[1],
                1.0,
                p,
                "isotropic",
                mu,
                mu_weight,
                psi_cos,
                psi_sin,
                sampled_mu,
                sampled_psi_cos,
                sampled_psi_sin);
            const int polar = (sampled_mu == mu[0]) ? 0 : 1;
            ++discrete_polar_counts[static_cast<std::size_t>(polar)];
            int azimuth = -1;
            for (int q = 0; q < 4; ++q) {
                if (sampled_psi_cos == psi_cos[static_cast<std::size_t>(q)]
                    && sampled_psi_sin == psi_sin[static_cast<std::size_t>(q)]) {
                    azimuth = q;
                    break;
                }
            }
            assert(azimuth >= 0);
            ++discrete_azimuth_counts[static_cast<std::size_t>(azimuth)];
        }
        assert(discrete_polar_counts[0] == 4
            && discrete_polar_counts[1] == 4);
        for (int count : discrete_azimuth_counts) {
            assert(count == 2);
        }
    }
    ev::Params yukawa = p;
    yukawa.bp_yukawa_mchi = 2.0;
    yukawa.bp_yukawa_mphi = 0.7;
    const double yukawa_speed = 1.3;
    const double yukawa_mphi2 =
        yukawa.bp_yukawa_mphi * yukawa.bp_yukawa_mphi;
    const double yukawa_scale =
        0.5 * yukawa.bp_yukawa_mchi * yukawa.bp_yukawa_mchi
        * yukawa_speed * yukawa_speed;
    for (double unit : {0.0, 0.2, 0.5, 0.8, 1.0}) {
        const double mu =
            ev::bp_sampled_scatter_mu_from_unit(
                unit, yukawa_speed, yukawa, "yukawa_born");
        const double inverse_mu =
            1.0 / (yukawa_mphi2 + yukawa_scale * (1.0 - mu));
        const double inverse_minus =
            1.0 / (yukawa_mphi2 + 2.0 * yukawa_scale);
        const double inverse_plus = 1.0 / yukawa_mphi2;
        const double recovered =
            (inverse_mu - inverse_minus)
            / (inverse_plus - inverse_minus);
        assert(std::abs(recovered - unit) < 2.0e-14);
    }
    ev::Params parsed = p;
    ev::apply_param(parsed, "bp_sampled_L_strata", "5", "sampled test");
    assert(parsed.bp_sampled_L_strata == 5);
    ev::validate_params(p);
    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    const double radius = geometry.radial.center[1];
    const std::vector<double>& w = geometry.radial_velocity.center;
    const std::vector<double>& eta = geometry.eta.center;
    std::vector<double> maxwellian(
        static_cast<std::size_t>(p.Nw * p.NL), 0.0);
    for (int q = 0; q < p.Nw * p.NL; ++q) {
        const double wr = ev::bp_reversible_radial_velocity(q, w, p);
        const double energy =
            ev::bp_reversible_energy(q, radius, w, eta, p);
        maxwellian[static_cast<std::size_t>(q)] =
            std::exp(0.12 * wr - energy / 0.9);
    }
    const ev::BPSampledProposal proposal =
        ev::bp_sampled_build_proposal(
            maxwellian, radius, w, eta, geometry, p);
    double proposal_sum = 0.0;
    double minimum_probability = std::numeric_limits<double>::infinity();
    for (int first = 0; first < p.Nw * p.NL; ++first) {
        for (int second = 0; second < p.Nw * p.NL; ++second) {
            for (int iphi = 0;
                 iphi < p.bp_boltzmann_pair_phi_points;
                 ++iphi) {
                const double probability =
                    ev::bp_sampled_joint_proposal_probability(
                        proposal, first, second, iphi);
                proposal_sum += probability;
                minimum_probability =
                    std::min(minimum_probability, probability);
            }
        }
    }
    assert(std::abs(proposal_sum - 1.0) < 2.0e-13);
    assert(minimum_probability > 0.0);
    std::vector<double> sparse(maxwellian.size(), 0.0);
    sparse[0] = 1.0;
    const ev::BPSampledProposal sparse_proposal =
        ev::bp_sampled_build_proposal(
            sparse, radius, w, eta, geometry, p);
    const int cells = p.Nw * p.NL;
    const double uniform_state_probability =
        p.bp_sampled_uniform_fraction
        / static_cast<double>(cells * cells)
        / static_cast<double>(p.bp_boltzmann_pair_phi_points);
    double expected_empty_pair_probability = 0.0;
    for (int first = 1; first < cells; ++first) {
        for (int second = 1; second < cells; ++second) {
            for (int iphi = 0;
                 iphi < p.bp_boltzmann_pair_phi_points;
                 ++iphi) {
                const double probability =
                    ev::bp_sampled_joint_proposal_probability(
                        sparse_proposal, first, second, iphi);
                const double roundoff = 8.0
                    * std::numeric_limits<double>::epsilon()
                    * uniform_state_probability;
                assert(probability + roundoff >= uniform_state_probability);
                expected_empty_pair_probability += probability;
            }
        }
    }
    constexpr std::uint64_t proposal_draws = UINT64_C(262144);
    std::uint64_t observed_empty_pair_draws = 0;
    for (std::uint64_t draw = 0; draw < proposal_draws; ++draw) {
        int first = 0;
        int second = 0;
        int iphi = 0;
        std::tie(first, second, iphi) =
            ev::bp_sampled_draw_joint_state(
                sparse_proposal, draw, UINT64_C(998877));
        (void)iphi;
        if (first > 0 && second > 0) {
            ++observed_empty_pair_draws;
        }
    }
    const double observed_empty_pair_probability =
        static_cast<double>(observed_empty_pair_draws)
        / static_cast<double>(proposal_draws);
    assert(std::abs(
        observed_empty_pair_probability - expected_empty_pair_probability)
        < 2.0e-3);
    for (int q = 0; q < p.Nw * p.NL; ++q) {
        const int reverse =
            proposal.time_reverse_cell[static_cast<std::size_t>(q)];
        assert(reverse >= 0);
        assert(proposal.time_reverse_cell[static_cast<std::size_t>(reverse)] == q);
    }
    const std::vector<int> dense_cells =
        ev::bp_sampled_dense_velocity_cells(
            proposal, radius, w, eta, p);
    for (int q : dense_cells) {
        const int reverse =
            proposal.time_reverse_cell[static_cast<std::size_t>(q)];
        assert(std::find(dense_cells.begin(), dense_cells.end(), reverse)
            != dense_cells.end());
    }

    std::vector<double> heat_first(maxwellian.size(), 0.0);
    std::vector<double> heat_second(maxwellian.size(), 0.0);
    const int heat_cell =
        ev::bp_reversible_velocity_index(p.Nw - 1, p.NL - 1, p);
    heat_first[static_cast<std::size_t>(heat_cell)] = 1.0;
    heat_second[static_cast<std::size_t>(heat_cell)] = -1.0;
    std::vector<double> cell_volume(maxwellian.size(), 0.0);
    for (int q = 0; q < p.Nw * p.NL; ++q) {
        const std::array<int, 2> indices =
            ev::bp_reversible_velocity_indices(q, p);
        cell_volume[static_cast<std::size_t>(q)] =
            2.0 * ev::PI
            * geometry.radial_velocity.width[
                static_cast<std::size_t>(indices[0])]
            * geometry.eta.width[static_cast<std::size_t>(indices[1])]
            / (radius * radius);
    }
    const ev::BPSampledBatchError cancellation_error =
        ev::bp_sampled_batch_error(
            heat_first,
            heat_second,
            cell_volume,
            radius,
            w,
            eta,
            p);
    assert(std::abs(cancellation_error.heat_gross - 1.0) < 1.0e-13);
    assert(std::abs(
        cancellation_error.heat
        - 1.0 / p.bp_sampled_heat_floor_fraction) < 1.0e-12);
    std::vector<double> gross_first(maxwellian.size(), 0.0);
    std::vector<double> gross_second(maxwellian.size(), 0.0);
    gross_first[static_cast<std::size_t>(heat_cell)] = 100.0;
    gross_second[static_cast<std::size_t>(heat_cell)] = 100.0;
    const ev::BPSampledBatchError rhs_floor_error =
        ev::bp_sampled_batch_error(
            heat_first,
            heat_second,
            cell_volume,
            radius,
            w,
            eta,
            p,
            &gross_first,
            &gross_second);
    assert(std::abs(rhs_floor_error.rhs_gross - 0.01) < 1.0e-14);
    assert(std::abs(
        rhs_floor_error.rhs
        - rhs_floor_error.rhs_gross
            / p.bp_sampled_rhs_floor_fraction) < 1.0e-14);

    for (double activity : {0.03125, 0.4, 2.0}) {
        double weighted_mean = 0.0;
        constexpr std::uint64_t ntc_trials = UINT64_C(65536);
        for (std::uint64_t draw = 0; draw < ntc_trials; ++draw) {
            const double unit =
                ev::bp_sampled_shifted_halton(
                    draw, 6, UINT64_C(918273645));
            weighted_mean +=
                activity
                * ev::bp_sampled_ntc_inverse_acceptance(
                    activity, 1.0, unit);
        }
        weighted_mean /= static_cast<double>(ntc_trials);
        assert(std::abs(weighted_mean - activity)
            < 5.0e-4 * activity);
    }

    ev::Params fixed = p;
    fixed.bp_sampled_min_samples_per_batch = 1024;
    fixed.bp_sampled_max_samples_per_batch = 1024;
    const ev::BPSampledShellGraph sampled_maxwellian =
        ev::bp_sampled_build_shell_graph(
            maxwellian,
            radius,
            w,
            eta,
            geometry,
            fixed,
            1024,
            9);
    assert(sampled_maxwellian.dense_velocity_cells
        == static_cast<int>(dense_cells.size()));
    const std::vector<double> maxwellian_rhs =
        ev::bp_reversible_shell_rhs(
            maxwellian, sampled_maxwellian.graph, fixed);
    double maxwellian_residual = 0.0;
    for (double value : maxwellian_rhs) {
        maxwellian_residual =
            std::max(maxwellian_residual, std::abs(value));
    }
    assert(maxwellian_residual < 5.0e-10);
    assert(sampled_maxwellian.complement_draws == 2 * 1024);

    ev::Params replicated = fixed;
    replicated.bp_sampled_replicates_per_estimate = 2;
    const ev::BPSampledShellGraph replicated_maxwellian =
        ev::bp_sampled_build_shell_graph(
            maxwellian,
            radius,
            w,
            eta,
            geometry,
            replicated,
            1024,
            9);
    const std::vector<double> replicated_maxwellian_rhs =
        ev::bp_reversible_shell_rhs(
            maxwellian, replicated_maxwellian.graph, replicated);
    double replicated_maxwellian_residual = 0.0;
    for (double value : replicated_maxwellian_rhs) {
        replicated_maxwellian_residual =
            std::max(replicated_maxwellian_residual, std::abs(value));
    }
    assert(replicated_maxwellian_residual < 5.0e-10);
    assert(replicated_maxwellian.complement_draws == 4 * 1024);

    std::vector<double> even_perturbation(maxwellian.size(), 0.0);
    for (int q = 0; q < p.Nw * p.NL; ++q) {
        const double energy =
            ev::bp_reversible_energy(q, radius, w, eta, p);
        even_perturbation[static_cast<std::size_t>(q)] =
            std::exp(-energy / 0.9)
            * (1.0 + 0.15 * energy);
    }
    const ev::BPSampledProposal even_proposal =
        ev::bp_sampled_build_proposal(
            even_perturbation, radius, w, eta, geometry, fixed);
    const std::vector<int> even_dense_cells =
        ev::bp_sampled_dense_velocity_cells(
            even_proposal, radius, w, eta, fixed);
    const ev::BPReversibleGraph even_dense_graph =
        ev::bp_sampled_build_dense_graph(
            even_perturbation,
            even_dense_cells,
            radius,
            w,
            eta,
            geometry,
            fixed);
    const std::vector<double> even_dense_rhs =
        ev::bp_reversible_shell_rhs(
            even_perturbation, even_dense_graph, fixed);
    double even_dense_asymmetry = 0.0;
    for (int q = 0; q < p.Nw * p.NL; ++q) {
        const int reverse =
            even_proposal.time_reverse_cell[static_cast<std::size_t>(q)];
        even_dense_asymmetry = std::max(
            even_dense_asymmetry,
            std::abs(
                even_dense_rhs[static_cast<std::size_t>(q)]
                - even_dense_rhs[static_cast<std::size_t>(reverse)]));
    }
    const ev::BPSampledShellGraph sampled_even =
        ev::bp_sampled_build_shell_graph(
            even_perturbation,
            radius,
            w,
            eta,
            geometry,
            fixed,
            1024,
            10);
    const std::vector<double> even_rhs =
        ev::bp_reversible_shell_rhs(
            even_perturbation, sampled_even.graph, fixed);
    double even_rhs_scale = 0.0;
    double even_rhs_asymmetry = 0.0;
    int even_rhs_asymmetry_cell = -1;
    for (int q = 0; q < p.Nw * p.NL; ++q) {
        const int reverse =
            proposal.time_reverse_cell[static_cast<std::size_t>(q)];
        even_rhs_scale = std::max(
            even_rhs_scale,
            std::abs(even_rhs[static_cast<std::size_t>(q)]));
        const double asymmetry = std::abs(
            even_rhs[static_cast<std::size_t>(q)]
            - even_rhs[static_cast<std::size_t>(reverse)]);
        if (asymmetry > even_rhs_asymmetry) {
            even_rhs_asymmetry = asymmetry;
            even_rhs_asymmetry_cell = q;
        }
    }
    std::cout << "Sampled even-state symmetry: max="
              << even_rhs_asymmetry
              << " scale=" << even_rhs_scale
              << " dense=" << even_dense_asymmetry
              << " cell=" << even_rhs_asymmetry_cell << '\n';
    if (even_rhs_asymmetry
        >= 2.0e-11 * std::max(1.0, even_rhs_scale)) {
        std::cout << "  dense cells:";
        for (int q : even_dense_cells) {
            std::cout << ' ' << q;
        }
        std::cout << '\n';
        for (int q = 0; q < p.Nw * p.NL; ++q) {
            const int reverse =
                even_proposal.time_reverse_cell[static_cast<std::size_t>(q)];
            std::cout << "  q=" << q << " reverse=" << reverse
                      << " sampled=" << even_rhs[static_cast<std::size_t>(q)]
                      << " dense="
                      << even_dense_rhs[static_cast<std::size_t>(q)] << '\n';
        }
    }
    assert(even_rhs_asymmetry
        < 2.0e-11 * std::max(1.0, even_rhs_scale));

    std::vector<double> perturbed = maxwellian;
    perturbed[static_cast<std::size_t>(
        ev::bp_reversible_velocity_index(0, p.NL - 1, p))] *= 1.8;
    perturbed[static_cast<std::size_t>(
        ev::bp_reversible_velocity_index(p.Nw - 1, p.NL - 1, p))] *= 0.65;
    ev::Params reference_params = p;
    reference_params.bp_collision_operator = "boltzmann_reversible";
    const ev::BPReversibleGraph reference =
        ev::bp_reversible_build_shell_graph(
            perturbed, radius, w, eta, geometry, reference_params);
    const std::vector<double> reference_rhs =
        ev::bp_reversible_shell_rhs(
            perturbed, reference, reference_params);

    auto build_at = [&](int samples) {
        ev::Params level = p;
        level.bp_sampled_min_samples_per_batch = samples;
        level.bp_sampled_max_samples_per_batch = samples;
        return ev::bp_sampled_build_shell_graph(
            perturbed,
            radius,
            w,
            eta,
            geometry,
            level,
            samples,
            11);
    };
    const ev::BPSampledShellGraph sampled_N = build_at(1024);
    const ev::BPSampledShellGraph sampled_2N = build_at(2048);
    const ev::BPSampledShellGraph sampled_4N = build_at(4096);
    auto rhs_error = [&](const ev::BPSampledShellGraph& sampled) {
        const std::vector<double> rhs =
            ev::bp_reversible_shell_rhs(perturbed, sampled.graph, p);
        double difference = 0.0;
        double scale = 0.0;
        for (std::size_t q = 0; q < rhs.size(); ++q) {
            const double volume = sampled.graph.velocity_cell_volume[q];
            difference += volume
                * (rhs[q] - reference_rhs[q])
                * (rhs[q] - reference_rhs[q]);
            scale += volume * reference_rhs[q] * reference_rhs[q];
        }
        return std::sqrt(difference / std::max(scale, 1.0e-300));
    };
    const double error_N = rhs_error(sampled_N);
    const double error_2N = rhs_error(sampled_2N);
    const double error_4N = rhs_error(sampled_4N);
    assert(error_4N < 1.25 * error_N);
    assert(std::min(error_2N, error_4N) < error_N);

    const std::vector<double> sampled_rhs =
        ev::bp_reversible_shell_rhs(
            perturbed, sampled_4N.graph, p);
    double mass_rate = 0.0;
    double momentum_rate = 0.0;
    double energy_rate = 0.0;
    double entropy_rate = 0.0;
    double energy_scale = 0.0;
    for (int q = 0; q < p.Nw * p.NL; ++q) {
        const double volume =
            sampled_4N.graph.velocity_cell_volume[
                static_cast<std::size_t>(q)];
        const double wr =
            ev::bp_reversible_radial_velocity(q, w, p);
        const double energy =
            ev::bp_reversible_energy(q, radius, w, eta, p);
        const double rate = sampled_rhs[static_cast<std::size_t>(q)];
        mass_rate += volume * rate;
        momentum_rate += volume * wr * rate;
        energy_rate += volume * energy * rate;
        energy_scale += volume * energy * std::abs(rate);
        entropy_rate += volume
            * std::log(perturbed[static_cast<std::size_t>(q)]) * rate;
    }
    assert(std::abs(mass_rate) < 2.0e-11);
    assert(std::abs(momentum_rate) < 2.0e-10);
    assert(std::abs(energy_rate)
        < 2.0e-11 * std::max(1.0, energy_scale));
    assert(entropy_rate <= 2.0e-10);

    const ev::BPSampledShellGraph repeated = build_at(4096);
    const std::vector<double> repeated_rhs =
        ev::bp_reversible_shell_rhs(perturbed, repeated.graph, p);
    assert(repeated_rhs == sampled_rhs);

    ev::Params threaded_sampling = p;
    threaded_sampling.threads = 4;
    threaded_sampling.bp_sampled_min_samples_per_batch = 4096;
    threaded_sampling.bp_sampled_max_samples_per_batch = 4096;
    const ev::BPSampledShellGraph threaded =
        ev::bp_sampled_build_shell_graph(
            perturbed,
            radius,
            w,
            eta,
            geometry,
            threaded_sampling,
            4096,
            11);
    const std::vector<double> threaded_rhs =
        ev::bp_reversible_shell_rhs(
            perturbed, threaded.graph, threaded_sampling);
    assert(threaded.complement_draws == sampled_4N.complement_draws);
    assert(threaded.retained_collision_draws
        == sampled_4N.retained_collision_draws);
    assert(threaded_rhs == sampled_rhs);

    ev::Params adaptive = p;
    adaptive.bp_sampled_min_samples_per_batch = 512;
    adaptive.bp_sampled_max_samples_per_batch = 2048;
    adaptive.bp_sampled_rhs_tolerance = 0.0;
    adaptive.bp_sampled_heat_tolerance = 0.0;
    adaptive.bp_sampled_fail_on_unconverged = false;
    adaptive.bp_sampled_ntc_pair_thinning = false;
    const ev::BPSampledShellGraph accumulated =
        ev::bp_sampled_build_shell_graph(
            perturbed,
            radius,
            w,
            eta,
            geometry,
            adaptive,
            512,
            17);
    assert(accumulated.samples_per_batch == 2048);
    assert(accumulated.adaptive_rounds == 2);
    assert(accumulated.complement_draws >= 2 * 2048);

    ev::Params ntc_fallback = p;
    ntc_fallback.bp_sampled_min_samples_per_batch = 64;
    ntc_fallback.bp_sampled_max_samples_per_batch = 64;
    ntc_fallback.bp_sampled_rhs_tolerance = 0.0;
    ntc_fallback.bp_sampled_heat_tolerance = 0.0;
    ntc_fallback.bp_sampled_fail_on_unconverged = false;
    ntc_fallback.bp_sampled_ntc_pair_thinning = true;
    ntc_fallback.bp_sampled_ntc_quantile = 1.0;
    const ev::BPSampledShellGraph fallback_graph =
        ev::bp_sampled_build_shell_graph(
            perturbed,
            radius,
            w,
            eta,
            geometry,
            ntc_fallback,
            64,
            18);
    assert(fallback_graph.ntc_fallback);

    const std::vector<double> accumulated_rhs =
        ev::bp_reversible_shell_rhs(
            perturbed, accumulated.graph, adaptive);
    ev::Params fixed_prefix = adaptive;
    fixed_prefix.bp_sampled_min_samples_per_batch = 2048;
    fixed_prefix.bp_sampled_max_samples_per_batch = 2048;
    const ev::BPSampledShellGraph uninterrupted =
        ev::bp_sampled_build_shell_graph(
            perturbed,
            radius,
            w,
            eta,
            geometry,
            fixed_prefix,
            2048,
            17);
    const std::vector<double> uninterrupted_rhs =
        ev::bp_reversible_shell_rhs(
            perturbed, uninterrupted.graph, fixed_prefix);
    double prefix_difference = 0.0;
    double prefix_scale = 0.0;
    for (std::size_t q = 0; q < accumulated_rhs.size(); ++q) {
        const double volume =
            accumulated.graph.velocity_cell_volume[q];
        prefix_difference += volume
            * (accumulated_rhs[q] - uninterrupted_rhs[q])
            * (accumulated_rhs[q] - uninterrupted_rhs[q]);
        prefix_scale += volume
            * uninterrupted_rhs[q] * uninterrupted_rhs[q];
    }
    assert(std::sqrt(
        prefix_difference / std::max(prefix_scale, 1.0e-300))
        < 5.0e-13);
    double accumulated_mass_rate = 0.0;
    double accumulated_energy_rate = 0.0;
    double accumulated_energy_scale = 0.0;
    for (int q = 0; q < p.Nw * p.NL; ++q) {
        const double volume =
            accumulated.graph.velocity_cell_volume[
                static_cast<std::size_t>(q)];
        const double energy =
            ev::bp_reversible_energy(q, radius, w, eta, p);
        const double rate =
            accumulated_rhs[static_cast<std::size_t>(q)];
        accumulated_mass_rate += volume * rate;
        accumulated_energy_rate += volume * energy * rate;
        accumulated_energy_scale += volume * energy * std::abs(rate);
    }
    assert(std::abs(accumulated_mass_rate) < 2.0e-11);
    assert(std::abs(accumulated_energy_rate)
        < 2.0e-11 * std::max(1.0, accumulated_energy_scale));

    ev::Params smoke = p;
    smoke.bp_sampled_min_samples_per_batch = 512;
    smoke.bp_sampled_max_samples_per_batch = 512;
    smoke.bp_collision_cfl = 0.05;
    smoke.bp_collision_max_substeps = 100;
    smoke.bp_reversible_active_shell_floor_fraction = 1.0e-8;
    std::vector<double> f(
        static_cast<std::size_t>(smoke.Nr) * smoke.Nw * smoke.NL, 0.0);
    for (int i = 0; i < smoke.Nr; ++i) {
        const double shell_radius =
            geometry.radial.center[static_cast<std::size_t>(i)];
        for (int j = 0; j < smoke.Nw; ++j) {
            for (int k = 0; k < smoke.NL; ++k) {
                const int q =
                    ev::bp_reversible_velocity_index(j, k, smoke);
                const double wr =
                    geometry.radial_velocity.center[static_cast<std::size_t>(j)];
                const double energy =
                    0.5 * wr * wr
                    + geometry.eta.center[static_cast<std::size_t>(k)]
                        / (shell_radius * shell_radius);
                f[ev::idx(i, j, k, smoke)] =
                    std::exp(-energy / 0.9)
                    * ((q == smoke.NL) ? 1.1 : 1.0)
                    * ((i == smoke.Nr - 1) ? 1.0e-20 : 1.0);
            }
        }
    }
    ev::Params guarded_plan_params = smoke;
    guarded_plan_params.bp_reversible_active_shell_floor_fraction = 0.5;
    guarded_plan_params.bp_boltzmann_remap_max_skipped_fraction = 0.5;
    const ev::BPSampledShellPlan guarded_plan =
        ev::bp_sampled_build_shell_plan(
            f,
            geometry.radial.center,
            geometry.radial_velocity.center,
            geometry.eta.center,
            geometry,
            guarded_plan_params);
    assert(guarded_plan.inactive.front() == 0);
    assert(guarded_plan.inactive.back() == 1);
    assert(std::isfinite(guarded_plan.minimum_mean_free_path));
    assert(guarded_plan.minimum_mean_free_path > 0.0);
    assert(
        guarded_plan.maximum_radial_cell_width_over_mean_free_path > 0.0);
    ev::Params rejected_locality = smoke;
    rejected_locality.bp_collision_locality_max_ratio = 1.0e-30;
    std::vector<double> rejected_locality_f = f;
    bool rejected_nonlocal_cell = false;
    try {
        ev::apply_boltzmann_poisson_collision_step(
            rejected_locality_f,
            nullptr,
            geometry.radial.center,
            geometry.radial_velocity.center,
            geometry.L,
            geometry.radial_velocity.width,
            geometry.legacy_Lq,
            1.0e-4,
            rejected_locality,
            ev::MpiContext{});
    } catch (const std::runtime_error& error) {
        rejected_nonlocal_cell =
            std::string(error.what()).find("Delta r / lambda_rms")
            != std::string::npos;
    }
    assert(rejected_nonlocal_cell);
    const ev::KineticInvariantLedger before =
        ev::evaluate_kinetic_invariant_ledger(f, geometry, smoke);
    const ev::MpiContext mpi;
    const ev::BPCollisionStepInfo info =
        ev::apply_boltzmann_poisson_collision_step(
            f,
            nullptr,
            geometry.radial.center,
            geometry.radial_velocity.center,
            geometry.L,
            geometry.radial_velocity.width,
            geometry.legacy_Lq,
            1.0e-4,
            smoke,
            mpi);
    const ev::KineticInvariantLedger after =
        ev::evaluate_kinetic_invariant_ledger(f, geometry, smoke);
    assert(ev::kinetic_relative_error(before.mass, after.mass) < 5.0e-13);
    assert(ev::kinetic_relative_error(
        before.kinetic_energy, after.kinetic_energy) < 5.0e-13);
    assert(info.sampled_max_samples_per_batch == 512);
    assert(info.sampled_complement_draws >= 2 * 512);
    assert(info.sampled_inactive_shells == 1);
    assert(std::isfinite(info.minimum_mean_free_path));
    assert(info.minimum_mean_free_path > 0.0);
    assert(info.maximum_radial_cell_width_over_mean_free_path > 0.0);
    assert(info.nonlocal_collision_cells == 0);
    assert(info.remap_skipped_inactive_event_density > 0.0);
    assert(
        info.remap_skipped_inactive_event_density
        / info.remap_attempted_event_density
        < smoke.bp_boltzmann_remap_max_skipped_fraction);
    std::cout
        << "Sampled reversible Boltzmann: proposal_sum=" << proposal_sum
        << " Maxwellian_residual=" << maxwellian_residual
        << " RHS errors N/2N/4N="
        << error_N << '/' << error_2N << '/' << error_4N
        << " batch_RHS=" << sampled_4N.rhs_relative_error
        << " batch_RHS_gross=" << sampled_4N.rhs_gross_error
        << " batch_heat=" << sampled_4N.heat_relative_error
        << " batch_heat_gross=" << sampled_4N.heat_gross_error
        << " energy_invariant=" << sampled_4N.energy_invariant_error
        << '\n';
}

void test_streamed_reversible_boltzmann() {
    ev::Params p = base_params();
    p.Nr = 3;
    p.Nw = 4;
    p.NL = 3;
    p.w_stretch = 2.5;
    p.r_min = 0.5;
    p.r_max = 2.0;
    p.w_max = 2.5;
    p.L_max = 2.5;
    p.bp_collision_operator = "boltzmann_reversible_sampled";
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 0.15;
    p.bp_boltzmann_pair_phi_points = 4;
    p.bp_boltzmann_scatter_mu_points = 2;
    p.bp_boltzmann_scatter_phi_points = 4;
    p.bp_sampled_outgoing_angle_samples =
        p.bp_boltzmann_scatter_mu_points
        * p.bp_boltzmann_scatter_phi_points;
    p.bp_reversible_candidate_radius = 2;
    p.bp_reversible_max_velocity_cells = p.Nw * p.NL;
    p.bp_reversible_projection_tolerance = 1.0e-10;
    p.bp_boltzmann_mass_tol = 0.0;
    p.bp_boltzmann_remap_max_skipped_fraction = 0.5;
    p.bp_sampled_dense_cells = 2;
    p.bp_sampled_energy_strata = 3;
    p.bp_sampled_L_strata = 2;
    p.bp_sampled_tail_fraction = 0.25;
    p.bp_sampled_stratified_fraction = 0.25;
    p.bp_sampled_uniform_fraction = 0.05;
    p.bp_sampled_tail_power = 1.5;
    p.bp_sampled_min_samples_per_batch = 1024;
    p.bp_sampled_max_samples_per_batch = 32768;
    p.bp_sampled_rhs_tolerance = 0.0;
    p.bp_sampled_heat_tolerance = 0.0;
    p.bp_sampled_fail_on_unconverged = false;
    p.bp_sampled_ntc_pair_thinning = false;
    p.bp_sampled_seed = UINT64_C(420424);

    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    const double radius = geometry.radial.center[1];
    const std::vector<double>& w = geometry.radial_velocity.center;
    const std::vector<double>& eta = geometry.eta.center;
    const int cells = p.Nw * p.NL;
    std::vector<double> maxwellian(static_cast<std::size_t>(cells), 0.0);
    for (int q = 0; q < cells; ++q) {
        const double wr = ev::bp_reversible_radial_velocity(q, w, p);
        const double energy =
            ev::bp_reversible_energy(q, radius, w, eta, p);
        maxwellian[static_cast<std::size_t>(q)] =
            std::exp(0.12 * wr - energy / 0.9);
    }
    const ev::BPStreamedShellContext maxwellian_context =
        ev::bp_streamed_make_shell_context(
            maxwellian, radius, w, eta, geometry, p);
    double proposal_sum = 0.0;
    for (int first = 0; first < cells; ++first) {
        for (int second = 0; second < cells; ++second) {
            for (int iphi = 0;
                 iphi < p.bp_boltzmann_pair_phi_points;
                 ++iphi) {
                proposal_sum += ev::bp_sampled_joint_proposal_probability(
                    maxwellian_context.proposal, first, second, iphi);
            }
        }
    }
    assert(std::abs(proposal_sum - 1.0) < 2.0e-13);

    ev::BPStreamedProjectionCache cache;
    const ev::BPStreamedOperatorEstimate maxwellian_estimate =
        ev::bp_streamed_estimate_shell_operator(
            maxwellian,
            maxwellian_context,
            w,
            eta,
            geometry,
            p,
            UINT64_C(1111),
            UINT64_C(2222),
            cache,
            4096,
            false);
    double maxwellian_residual = 0.0;
    double maxwellian_gross = 0.0;
    for (std::size_t q = 0; q < maxwellian.size(); ++q) {
        maxwellian_residual = std::max(
            maxwellian_residual, std::abs(maxwellian_estimate.rhs[q]));
        maxwellian_gross = std::max(
            maxwellian_gross, maxwellian_estimate.gross_rhs[q]);
    }
    assert(maxwellian_residual
        < 2.0e-10 * std::max(1.0, maxwellian_gross));

    std::vector<double> perturbed = maxwellian;
    perturbed[static_cast<std::size_t>(
        ev::bp_reversible_velocity_index(0, p.NL - 1, p))] *= 1.8;
    perturbed[static_cast<std::size_t>(
        ev::bp_reversible_velocity_index(p.Nw - 1, p.NL - 1, p))] *= 0.65;
    const ev::BPStreamedShellContext context =
        ev::bp_streamed_make_shell_context(
            perturbed, radius, w, eta, geometry, p);

    cache.clear();
    const ev::BPStreamedOperatorEstimate estimate =
        ev::bp_streamed_estimate_shell_operator(
            perturbed,
            context,
            w,
            eta,
            geometry,
            p,
            UINT64_C(3333),
            UINT64_C(4444),
            cache,
            32768,
            false);
    double mass_rate = 0.0;
    double momentum_rate = 0.0;
    double energy_rate = 0.0;
    double mass_scale = 0.0;
    double momentum_scale = 0.0;
    double energy_scale = 0.0;
    for (int q = 0; q < cells; ++q) {
        const double volume = context.volume[static_cast<std::size_t>(q)];
        const double wr = ev::bp_reversible_radial_velocity(q, w, p);
        const double energy =
            ev::bp_reversible_energy(q, radius, w, eta, p);
        const double rate = estimate.rhs[static_cast<std::size_t>(q)];
        mass_rate += volume * rate;
        momentum_rate += volume * wr * rate;
        energy_rate += volume * energy * rate;
        mass_scale += volume * std::abs(rate);
        momentum_scale += volume * std::abs(wr * rate);
        energy_scale += volume * std::abs(energy * rate);
    }
    assert(std::abs(mass_rate)
        < 2.0e-11 * std::max(1.0, mass_scale));
    assert(std::abs(momentum_rate)
        < 2.0e-10 * std::max(1.0, momentum_scale));
    assert(std::abs(energy_rate)
        < 2.0e-10 * std::max(1.0, energy_scale));

    ev::Params exact_params = p;
    exact_params.bp_sampled_dense_cells = cells;
    const ev::BPStreamedShellContext exact_context =
        ev::bp_streamed_make_shell_context(
            perturbed, radius, w, eta, geometry, exact_params);
    std::cout << "Streamed all-dense complement: "
              << exact_context.complement_probability << std::endl;
    assert(exact_context.complement_probability < 1.0e-12);
    const std::vector<double>& exact_rhs = exact_context.dense_rhs;

    // Sum the expectation of the importance-sampled estimator over its
    // complete finite sample space.  This isolates estimator normalization
    // from finite-sample variance.
    ev::BPStreamedBatch exact_streamed;
    exact_streamed.rhs_sum = context.dense_rhs;
    exact_streamed.gross_rhs_sum = context.dense_gross_rhs;
    ev::BPStreamedProjectionCache exact_cache;
    for (int first = 0; first < cells; ++first) {
        for (int second = 0; second < cells; ++second) {
            for (int iphi = 0;
                 iphi < p.bp_boltzmann_pair_phi_points;
                 ++iphi) {
                if (context.dense_mask[static_cast<std::size_t>(first)]
                    && context.dense_mask[static_cast<std::size_t>(second)]) {
                    continue;
                }
                const double probability =
                    ev::bp_sampled_joint_proposal_probability(
                        context.proposal, first, second, iphi);
                const int reverse_first = context.proposal.time_reverse_cell[
                    static_cast<std::size_t>(first)];
                const int reverse_second = context.proposal.time_reverse_cell[
                    static_cast<std::size_t>(second)];
                const double reverse_probability =
                    ev::bp_sampled_joint_proposal_probability(
                        context.proposal,
                        reverse_first,
                        reverse_second,
                        iphi);
                const double orbit_probability =
                    probability + reverse_probability;
                int canonical_first = first;
                int canonical_second = second;
                int canonical_reverse_first = reverse_first;
                int canonical_reverse_second = reverse_second;
                if (std::tie(
                        canonical_reverse_first,
                        canonical_reverse_second)
                    < std::tie(canonical_first, canonical_second)) {
                    std::swap(canonical_first, canonical_reverse_first);
                    std::swap(canonical_second, canonical_reverse_second);
                }
                ev::BPFAVec3 center_of_mass;
                ev::BPFAVec3 incoming_direction;
                std::array<ev::BPFAVec3, 2> basis;
                double relative_speed = 0.0;
                double sigma = 0.0;
                if (!ev::bp_sampled_pair_kinematics(
                        canonical_first,
                        canonical_second,
                        iphi,
                        radius,
                        w,
                        eta,
                        context.pair_phi_cos,
                        context.pair_phi_sin,
                        p,
                        center_of_mass,
                        incoming_direction,
                        basis,
                        relative_speed,
                        sigma)) {
                    continue;
                }
                double polar_sum = 0.0;
                for (std::size_t im = 0;
                     im < context.scatter_mu.size();
                     ++im) {
                    polar_sum += context.scatter_mu_weight[im]
                        * ev::bp_fa_scatter_weight_raw(
                            context.scatter_mu[im],
                            relative_speed,
                            p,
                            context.kernel);
                }
                for (int im = 0;
                     im < static_cast<int>(context.scatter_mu.size());
                     ++im) {
                    const double polar_probability =
                        context.scatter_mu_weight[
                            static_cast<std::size_t>(im)]
                        * ev::bp_fa_scatter_weight_raw(
                            context.scatter_mu[static_cast<std::size_t>(im)],
                            relative_speed,
                            p,
                            context.kernel)
                        / polar_sum;
                    for (int ipsi = 0;
                         ipsi
                            < static_cast<int>(context.scatter_phi_cos.size());
                         ++ipsi) {
                        ev::BPFAVec3 outgoing_first;
                        ev::BPFAVec3 outgoing_second;
                        ev::bp_sampled_outgoing_velocities(
                            center_of_mass,
                            incoming_direction,
                            basis,
                            relative_speed,
                            context.scatter_mu[static_cast<std::size_t>(im)],
                            context.scatter_phi_cos[
                                static_cast<std::size_t>(ipsi)],
                            context.scatter_phi_sin[
                                static_cast<std::size_t>(ipsi)],
                            outgoing_first,
                            outgoing_second);
                        const ev::BPStreamedProjectionRecord& projection =
                            ev::bp_streamed_projection_record(
                                canonical_first,
                                canonical_second,
                                iphi,
                                im,
                                ipsi,
                                outgoing_first,
                                outgoing_second,
                                w,
                                eta,
                                geometry,
                                p,
                                context,
                                exact_cache);
                        const ev::BPStreamedProjectionRecord&
                            reverse_projection =
                                ev::bp_streamed_radial_reverse_projection_record(
                                    projection,
                                    canonical_reverse_first,
                                    canonical_reverse_second,
                                    iphi,
                                    im,
                                    ipsi,
                                    exact_cache);
                        const double angular_probability = polar_probability
                            / static_cast<double>(
                                context.scatter_phi_cos.size());
                        const double common =
                            0.25 * relative_speed * sigma
                            / static_cast<double>(
                                p.bp_boltzmann_pair_phi_points)
                            * probability * angular_probability
                            / orbit_probability;
                        if (projection.represented) {
                            ev::bp_streamed_accumulate_reaction(
                                perturbed,
                                canonical_first,
                                canonical_second,
                                projection.direct_species,
                                common
                                    * context.volume[
                                        static_cast<std::size_t>(canonical_first)]
                                    * context.volume[
                                        static_cast<std::size_t>(canonical_second)],
                                context.volume,
                                p.bp_reversible_entropy_floor,
                                exact_streamed);
                        }
                        if (reverse_projection.represented) {
                            ev::bp_streamed_accumulate_reaction(
                                perturbed,
                                canonical_reverse_first,
                                canonical_reverse_second,
                                reverse_projection.direct_species,
                                common
                                    * context.volume[
                                        static_cast<std::size_t>(
                                            canonical_reverse_first)]
                                    * context.volume[
                                        static_cast<std::size_t>(
                                            canonical_reverse_second)],
                                context.volume,
                                p.bp_reversible_entropy_floor,
                                exact_streamed);
                        }
                    }
                }
            }
        }
    }
    double exact_expectation_difference = 0.0;
    double exact_expectation_scale = 0.0;
    for (int q = 0; q < cells; ++q) {
        const double volume = context.volume[static_cast<std::size_t>(q)];
        const double delta = exact_streamed.rhs_sum[static_cast<std::size_t>(q)]
            - exact_rhs[static_cast<std::size_t>(q)];
        exact_expectation_difference += volume * delta * delta;
        exact_expectation_scale += volume
            * exact_rhs[static_cast<std::size_t>(q)]
            * exact_rhs[static_cast<std::size_t>(q)];
    }
    const double exact_expectation_error = std::sqrt(
        exact_expectation_difference
        / std::max(exact_expectation_scale, 1.0e-300));
    std::cout << "Streamed exact expectation error: "
              << exact_expectation_error << std::endl;
    assert(exact_expectation_error < 2.0e-12);

    auto relative_rhs_error = [&](int samples, std::uint64_t salt) {
        ev::BPStreamedProjectionCache local_cache;
        const ev::BPStreamedOperatorEstimate sampled =
            ev::bp_streamed_estimate_shell_operator(
                perturbed,
                context,
                w,
                eta,
                geometry,
                p,
                UINT64_C(51001) ^ salt,
                UINT64_C(81013) ^ salt,
                local_cache,
                samples,
                false);
        double difference = 0.0;
        double scale = 0.0;
        for (int q = 0; q < cells; ++q) {
            const double volume = context.volume[static_cast<std::size_t>(q)];
            const double delta = sampled.rhs[static_cast<std::size_t>(q)]
                - exact_rhs[static_cast<std::size_t>(q)];
            difference += volume * delta * delta;
            scale += volume
                * exact_rhs[static_cast<std::size_t>(q)]
                * exact_rhs[static_cast<std::size_t>(q)];
        }
        return std::sqrt(difference / std::max(scale, 1.0e-300));
    };
    const double error_N = relative_rhs_error(2048, UINT64_C(7));
    const double error_4N = relative_rhs_error(8192, UINT64_C(7));
    const double error_16N = relative_rhs_error(32768, UINT64_C(7));
    std::cout << "Streamed collision convergence: "
              << error_N << " " << error_4N << " " << error_16N
              << " exact=" << exact_expectation_error
              << " cache_hits=" << cache.hits
              << " cache_misses=" << cache.misses << '\n';
    assert(error_16N < error_N);
    assert(error_16N < 0.12);

    ev::Params production_adaptive = p;
    production_adaptive.bp_sampled_min_samples_per_batch = 256;
    production_adaptive.bp_sampled_max_samples_per_batch = 2048;
    production_adaptive.bp_sampled_rhs_tolerance = 1.0e-5;
    production_adaptive.bp_sampled_heat_tolerance = 1.0e-5;
    const ev::BPStreamedShellContext production_context =
        ev::bp_streamed_make_shell_context(
            perturbed,
            radius,
            w,
            eta,
            geometry,
            production_adaptive);
    ev::BPStreamedProjectionCache production_cache;
    const ev::BPStreamedOperatorEstimate production_estimate =
        ev::bp_streamed_estimate_shell_operator(
            perturbed,
            production_context,
            w,
            eta,
            geometry,
            production_adaptive,
            UINT64_C(61001),
            UINT64_C(91009),
            production_cache,
            256,
            true);
    assert(production_estimate.converged
        || production_estimate.samples_per_batch
            == production_adaptive.bp_sampled_max_samples_per_batch);

    ev::Params parity_params = p;
    parity_params.bp_sampled_outgoing_angle_samples =
        parity_params.bp_boltzmann_scatter_mu_points
        * parity_params.bp_boltzmann_scatter_phi_points;
    parity_params.bp_sampled_ntc_pair_thinning = false;
    std::vector<double> even_state(maxwellian.size(), 0.0);
    for (int q = 0; q < cells; ++q) {
        const double energy =
            ev::bp_reversible_energy(q, radius, w, eta, parity_params);
        even_state[static_cast<std::size_t>(q)] =
            std::exp(-energy / 0.9) * (1.0 + 0.15 * energy);
    }
    const ev::BPStreamedShellContext even_context =
        ev::bp_streamed_make_shell_context(
            even_state, radius, w, eta, geometry, parity_params);
    ev::BPStreamedProjectionCache even_cache;
    const ev::BPStreamedOperatorEstimate even_estimate =
        ev::bp_streamed_estimate_shell_operator(
            even_state,
            even_context,
            w,
            eta,
            geometry,
            parity_params,
            UINT64_C(91919),
            UINT64_C(92929),
            even_cache,
            2048,
            false);
    double odd_heat = 0.0;
    double odd_heat_scale = 0.0;
    for (int q = 0; q < cells; ++q) {
        const double volume = even_context.volume[static_cast<std::size_t>(q)];
        const double wr = ev::bp_reversible_radial_velocity(q, w, parity_params);
        const double energy = ev::bp_reversible_energy(
            q, radius, w, eta, parity_params);
        const double contribution = volume * wr * energy
            * even_estimate.rhs[static_cast<std::size_t>(q)];
        odd_heat += contribution;
        odd_heat_scale += std::abs(contribution);
    }
    std::cout << "Streamed even-state odd heat: "
              << odd_heat << " scale=" << odd_heat_scale << std::endl;
    assert(std::abs(odd_heat)
        < 2.0e-11 * std::max(1.0, odd_heat_scale));

    ev::Params smoke = p;
    smoke.bp_collision_operator = "boltzmann_reversible_streamed";
    smoke.bp_sampled_min_samples_per_batch = 2048;
    smoke.bp_sampled_max_samples_per_batch = 2048;
    smoke.bp_sampled_rhs_tolerance = 1.0;
    smoke.bp_sampled_heat_tolerance = 1.0;
    smoke.bp_sampled_fail_on_unconverged = false;
    smoke.bp_collision_cfl = 0.05;
    smoke.bp_collision_max_substeps = 20;
    smoke.bp_collision_fail_on_nonlocal_cells = false;
    smoke.bp_sampled_operator_refresh_l1_fraction = 2.0;
    ev::validate_params(smoke);
    for (double invalid_threshold : {
             -1.0e-12,
             std::numeric_limits<double>::infinity()}) {
        ev::Params invalid_refresh = smoke;
        invalid_refresh.bp_sampled_operator_refresh_l1_fraction =
            invalid_threshold;
        bool rejected_refresh = false;
        try {
            ev::validate_params(invalid_refresh);
        } catch (const std::runtime_error& error) {
            rejected_refresh =
                std::string(error.what()).find("refresh L1 fraction")
                != std::string::npos;
        }
        assert(rejected_refresh);
    }
    const std::uint64_t base_operator_hash =
        ev::bp_streamed_operator_configuration_hash(smoke);
    ev::Params changed_sequence = smoke;
    changed_sequence.bp_sampled_fixed_geometric_sequence = false;
    assert(ev::bp_streamed_operator_configuration_hash(changed_sequence)
        != base_operator_hash);
    ev::Params changed_reuse = smoke;
    changed_reuse.bp_sampled_reuse_operator_graph = false;
    assert(ev::bp_streamed_operator_configuration_hash(changed_reuse)
        != base_operator_hash);
    ev::Params changed_refresh = smoke;
    changed_refresh.bp_sampled_operator_refresh_l1_fraction = 1.0;
    assert(ev::bp_streamed_operator_configuration_hash(changed_refresh)
        != base_operator_hash);
    ev::Params changed_softened_gravity = smoke;
    changed_softened_gravity.bp_subhalo_mass *= 2.0;
    assert(ev::bp_streamed_operator_configuration_hash(
               changed_softened_gravity)
        != base_operator_hash);
    ev::Params changed_gravity = smoke;
    changed_gravity.vp_gravity *= 2.0;
    assert(ev::bp_streamed_operator_configuration_hash(changed_gravity)
        != base_operator_hash);
    ev::Params changed_entropy_floor = smoke;
    changed_entropy_floor.bp_reversible_entropy_floor *= 10.0;
    assert(ev::bp_streamed_operator_configuration_hash(
               changed_entropy_floor)
        != base_operator_hash);
    ev::Params changed_failure_contract = smoke;
    changed_failure_contract.bp_sampled_fail_on_unconverged = true;
    assert(ev::bp_streamed_operator_configuration_hash(
               changed_failure_contract)
        != base_operator_hash);
    std::vector<double> full_f(
        static_cast<std::size_t>(smoke.Nr * smoke.Nw * smoke.NL), 0.0);
    for (int i = 0; i < smoke.Nr; ++i) {
        const double shell_radius =
            geometry.radial.center[static_cast<std::size_t>(i)];
        for (int q = 0; q < cells; ++q) {
            const std::array<int, 2> indices =
                ev::bp_reversible_velocity_indices(q, smoke);
            const double wr = w[static_cast<std::size_t>(indices[0])];
            const double energy = 0.5 * wr * wr
                + eta[static_cast<std::size_t>(indices[1])]
                    / (shell_radius * shell_radius);
            full_f[ev::idx(i, indices[0], indices[1], smoke)] =
                std::exp(-energy / 0.9)
                * ((q == ev::bp_reversible_velocity_index(
                        0, smoke.NL - 1, smoke))
                    ? 1.2
                    : 1.0);
        }
    }
    const ev::KineticInvariantLedger before =
        ev::evaluate_kinetic_invariant_ledger(full_f, geometry, smoke);
    ev::BPStreamedCollisionWorkspace persistent_workspace;
    const ev::BPCollisionStepInfo collision =
        ev::apply_boltzmann_poisson_collision_step(
            full_f,
            nullptr,
            geometry.radial.center,
            geometry.radial_velocity.center,
            geometry.L,
            geometry.radial_velocity.width,
            geometry.legacy_Lq,
            1.0e-5,
            smoke,
            ev::MpiContext{},
            &persistent_workspace);
    const ev::KineticInvariantLedger after =
        ev::evaluate_kinetic_invariant_ledger(full_f, geometry, smoke);
    assert(ev::kinetic_relative_error(before.mass, after.mass) < 2.0e-12);
    assert(ev::kinetic_relative_error(
        before.radial_momentum, after.radial_momentum) < 2.0e-11);
    assert(ev::kinetic_relative_error(
        before.kinetic_energy, after.kinetic_energy) < 2.0e-11);
    assert(collision.sampled_complement_draws > 0);
    assert(collision.sampled_retained_collision_draws > 0);
    assert(collision.sampled_reused_operator_shells == 0);
    assert(!persistent_workspace.shell_operator_graph_cache.empty());
    int cached_operator_shells = 0;
    for (ev::BPStreamedOperatorGraphCache& cache :
         persistent_workspace.shell_operator_graph_cache) {
        if (cache.valid) {
            // Exercise the reuse diagnostic independently of whether this
            // deliberately loose smoke estimate happened to converge.
            cache.converged = false;
            ++cached_operator_shells;
        }
    }
    assert(cached_operator_shells > 0);

    const ev::KineticInvariantLedger before_reuse =
        ev::evaluate_kinetic_invariant_ledger(full_f, geometry, smoke);
    const ev::BPCollisionStepInfo reused_collision =
        ev::apply_boltzmann_poisson_collision_step(
            full_f,
            nullptr,
            geometry.radial.center,
            geometry.radial_velocity.center,
            geometry.L,
            geometry.radial_velocity.width,
            geometry.legacy_Lq,
            1.0e-5,
            smoke,
            ev::MpiContext{},
            &persistent_workspace);
    const ev::KineticInvariantLedger after_reuse =
        ev::evaluate_kinetic_invariant_ledger(full_f, geometry, smoke);
    assert(reused_collision.sampled_reused_operator_shells > 0);
    assert(reused_collision.sampled_unconverged_shells
        >= reused_collision.sampled_reused_operator_shells);
    assert(ev::kinetic_relative_error(
        before_reuse.mass, after_reuse.mass) < 2.0e-12);
    assert(ev::kinetic_relative_error(
        before_reuse.radial_momentum, after_reuse.radial_momentum)
        < 2.0e-11);
    assert(ev::kinetic_relative_error(
        before_reuse.kinetic_energy, after_reuse.kinetic_energy)
        < 2.0e-11);
}

void test_implicit_fractional_reverse_reaction() {
    ev::Params p = base_params();
    p.bp_reversible_entropy_floor = 1.0e-300;
    p.bp_reversible_entropy_tolerance = 1.0e-12;
    ev::BPReversibleGraph graph;
    graph.velocity_cell_volume.assign(6, 1.0);
    const std::vector<ev::BPReversibleProductPair> products{
        ev::BPReversibleProductPair{2, 3, 0.5},
        ev::BPReversibleProductPair{4, 5, 0.5},
    };
    ev::BPReversibleReaction reaction;
    reaction.incoming_first = 0;
    reaction.incoming_second = 1;
    reaction.coefficient = 1.0;
    reaction.species =
        ev::bp_reversible_compress_reaction(0, 1, products);
    graph.reactions.push_back(reaction);

    const std::vector<double> initial{
        1.0e-80,
        1.0e-80,
        1.0e-120,
        1.0,
        1.0,
        1.0,
    };
    const std::vector<double> rhs =
        ev::bp_reversible_shell_rhs(initial, graph, p);
    const double explicit_limit =
        ev::bp_reversible_positive_dt_limit(initial, rhs, 0.1);
    assert(explicit_limit < 1.0e-50);
    const double mass_before =
        std::accumulate(initial.begin(), initial.end(), 0.0);
    const double entropy_before =
        ev::bp_reversible_shell_entropy(
            initial, graph, p.bp_reversible_entropy_floor);
    const ev::BPReversibleShellResult advanced =
        ev::bp_reversible_advance_shell(initial, graph, 1.0e-3, p);
    assert(ev::bp_reversible_nonnegative(advanced.f));
    assert(advanced.reaction_updates == 2);
    assert(advanced.implicit_reaction_updates > 0);
    assert(advanced.implicit_iterations > 0);
    const double mass_after =
        std::accumulate(advanced.f.begin(), advanced.f.end(), 0.0);
    const double entropy_after =
        ev::bp_reversible_shell_entropy(
            advanced.f, graph, p.bp_reversible_entropy_floor);
    assert(ev::kinetic_relative_error(mass_before, mass_after) < 1.0e-14);
    assert(entropy_after <= entropy_before
        + 1.0e-12 * std::max(1.0, std::abs(entropy_before)));

    const std::vector<double> maxwellian(6, 1.0);
    const ev::BPReversibleShellResult equilibrium =
        ev::bp_reversible_advance_shell(maxwellian, graph, 1.0, p);
    for (double value : equilibrium.f) {
        assert(std::abs(value - 1.0) < 1.0e-15);
    }

    ev::Params fallback_p = p;
    fallback_p.bp_reversible_explicit_fast_path = true;
    fallback_p.bp_collision_max_substeps = 0;
    const ev::BPReversibleShellResult fast_path_fallback =
        ev::bp_reversible_advance_shell(
            maxwellian, graph, 1.0, fallback_p);
    assert(fast_path_fallback.explicit_fast_path_fallbacks == 1);
    assert(fast_path_fallback.reaction_updates == 2);
    for (double value : fast_path_fallback.f) {
        assert(std::abs(value - 1.0) < 1.0e-15);
    }

    ev::BPReversibleGraph subdouble_graph = graph;
    subdouble_graph.velocity_cell_volume[2] = 1.0e-200;
    std::vector<double> subdouble = initial;
    subdouble[2] = 1.0e-200;
    const ev::BPReversibleShellResult subdouble_advanced =
        ev::bp_reversible_advance_shell(
            subdouble, subdouble_graph, 1.0e-3, p);
    assert(ev::bp_reversible_nonnegative(subdouble_advanced.f));

    ev::BPReversibleGraph narrow_graph;
    narrow_graph.velocity_cell_volume.assign(4, 1.0);
    ev::BPReversibleReaction narrow_reaction;
    narrow_reaction.incoming_first = 0;
    narrow_reaction.incoming_second = 1;
    narrow_reaction.coefficient = 0.00460547;
    narrow_reaction.species = ev::bp_reversible_compress_reaction(
        0,
        1,
        {ev::BPReversibleProductPair{2, 3, 1.0}});
    narrow_graph.reactions.push_back(narrow_reaction);
    const std::vector<double> narrow_initial{
        0.0,
        0.0,
        7.71354e-32,
        2.34016e16,
    };
    const std::pair<double, double> narrow_bounds =
        ev::bp_reversible_extent_bounds(
            narrow_initial,
            narrow_reaction,
            narrow_graph.velocity_cell_volume);
    const double narrow_extent =
        ev::bp_reversible_reaction_extent(
            narrow_initial,
            narrow_reaction,
            narrow_graph.velocity_cell_volume,
            2.25e-5,
            p.bp_reversible_entropy_floor);
    assert(narrow_extent >= narrow_bounds.first);
    assert(narrow_extent <= narrow_bounds.second);
    const ev::BPReversibleReactionEvaluation narrow_evaluation =
        ev::bp_reversible_evaluate_reaction(
            narrow_initial,
            narrow_reaction,
            narrow_graph.velocity_cell_volume,
            narrow_extent,
            p.bp_reversible_entropy_floor);
    assert(narrow_evaluation.valid);
    const double narrow_residual =
        narrow_extent - 2.25e-5 * narrow_evaluation.flux;
    const double narrow_euler_scale =
        2.25e-5 * std::abs(
            ev::bp_reversible_evaluate_reaction(
                narrow_initial,
                narrow_reaction,
                narrow_graph.velocity_cell_volume,
                0.0,
                p.bp_reversible_entropy_floor).flux);
    std::cout
        << "Narrow implicit bracket: lower=" << narrow_bounds.first
        << " extent=" << narrow_extent
        << " residual=" << narrow_residual
        << " euler_scale=" << narrow_euler_scale
        << '\n';
    assert(std::abs(narrow_residual)
        <= 1024.0 * std::numeric_limits<double>::epsilon()
            * narrow_euler_scale);
    std::vector<double> narrow_updated = narrow_initial;
    ev::bp_reversible_apply_extent(
        narrow_updated,
        narrow_reaction,
        narrow_graph.velocity_cell_volume,
        narrow_extent);
    assert(ev::bp_reversible_nonnegative(narrow_updated));
}

void test_validated_strang_step() {
    ev::Params p = base_params();
    p.physics_model = "boltzmann_poisson";
    p.vp_orbit_regularization = "auto";
    p.Nr = 3;
    p.Nw = 5;
    p.NL = 2;
    p.r_min = 0.5;
    p.r_max = 2.0;
    p.w_max = 2.0;
    p.L_max = 2.0;
    p.kinetic_eta_power = 6.0;
    p.bp_collision_operator = "boltzmann_reversible";
    p.bp_collision_splitting = "strang";
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 0.01;
    p.bp_sigma0 = 0.01;
    p.bp_boltzmann_pair_phi_points = 4;
    p.bp_boltzmann_scatter_mu_points = 2;
    p.bp_boltzmann_scatter_phi_points = 4;
    p.bp_boltzmann_mass_tol = 0.0;
    p.bp_boltzmann_remap_max_skipped_fraction = 1.0e-6;
    p.bp_reversible_candidate_radius = 2;
    p.bp_reversible_max_velocity_cells = p.Nw * p.NL;
    p.bp_collision_cfl = 0.1;
    p.bp_collision_max_substeps = 100;
    p.beta = 6.0;
    ev::validate_params(p);

    const ev::PhaseSpaceCellGeometry geometry =
        ev::make_cell_average_eta_geometry(
            ev::make_positive_radial_cell_axis(p), p);
    const std::vector<double>& r = geometry.radial.center;
    const std::vector<double>& w = geometry.radial_velocity.center;
    const std::vector<double>& L = geometry.L;
    const std::vector<double> x_r = ev::radial_coordinate_grid(r, p);
    const ev::VlasovPoissonField seed_field =
        represented_test_field(geometry, p);
    std::vector<double> f = represented_energy_equilibrium(
        geometry, p, seed_field, 1.2, 0.6, 1.8);
    const ev::MpiContext mpi;
    constexpr double dt = 1.0e-7;

    const ev::KineticInvariantLedger initial =
        ev::evaluate_kinetic_invariant_ledger(f, geometry, p);
    assert(initial.mass > 0.0);
    const ev::BPCollisionStepInfo first_collision =
        ev::apply_bp_reversible_boltzmann_collision_step(
            f,
            nullptr,
            r,
            w,
            L,
            geometry.radial_velocity.width,
            geometry.legacy_Lq,
            0.5 * dt,
            p,
            mpi);
    const ev::KineticInvariantLedger after_first_collision =
        ev::evaluate_kinetic_invariant_ledger(f, geometry, p);
    assert(first_collision.explicit_dt_limit > 0.51 * dt);
    assert(ev::kinetic_relative_error(
        initial.mass, after_first_collision.mass) < 5.0e-13);
    assert(ev::kinetic_relative_error(
        initial.kinetic_energy,
        after_first_collision.kinetic_energy) < 5.0e-13);

    const ev::Moments moments = ev::compute_vp_moments(
        f,
        r,
        w,
        L,
        geometry.radial_velocity.width,
        geometry.legacy_Lq,
        p);
    const ev::VlasovPoissonField field =
        ev::solve_vlasov_poisson_field(moments, r, 0.0, p);
    const ev::SLDGMidpointPrediction midpoint_prediction =
        ev::predict_vlasov_poisson_midpoint_moments(
            moments, geometry, 0.5 * dt, p);
    const auto midpoint_remap =
        ev::advect_vlasov_poisson_conservative_sldg_frozen(
            f, x_r, r, w, L, field, geometry, 0.5 * dt, p);
    const ev::Moments midpoint_remap_moments = ev::compute_vp_moments(
        midpoint_remap.first,
        r,
        w,
        L,
        geometry.radial_velocity.width,
        geometry.legacy_Lq,
        p);
    double midpoint_density_difference = 0.0;
    double midpoint_density_scale = 0.0;
    for (int i = 0; i < p.Nr; ++i) {
        const double measure =
            4.0 * ev::PI
            * r[static_cast<std::size_t>(i)]
            * r[static_cast<std::size_t>(i)]
            * geometry.radial.width[static_cast<std::size_t>(i)];
        midpoint_density_difference += measure * std::abs(
            midpoint_prediction.moments.rho[static_cast<std::size_t>(i)]
            - midpoint_remap_moments.rho[static_cast<std::size_t>(i)]);
        midpoint_density_scale += measure
            * midpoint_remap_moments.rho[static_cast<std::size_t>(i)];
    }
    const double midpoint_density_relative_difference =
        midpoint_density_difference / midpoint_density_scale;
    assert(midpoint_density_relative_difference < 1.0e-5);
    assert(std::abs(
        midpoint_prediction.mass_before
        - midpoint_prediction.mass_after
        - midpoint_prediction.escaped_mass)
        < 1.0e-13 * std::max(1.0, midpoint_prediction.mass_before));
    const ev::SLDGStepResult transport =
        ev::advect_vlasov_poisson_conservative_sldg(
            f,
            x_r,
            r,
            w,
            L,
            field,
            moments,
            geometry,
            dt,
            0.0,
            p,
            mpi);
    f = transport.f;
    const ev::KineticInvariantLedger after_transport =
        ev::evaluate_kinetic_invariant_ledger(f, geometry, p);
    const double transport_loss =
        after_first_collision.mass - after_transport.mass;
    assert(std::abs(transport_loss - transport.diagnostics.escaped_mass)
        < 1.0e-11 * std::max(1.0, after_first_collision.mass));

    const ev::BPCollisionStepInfo second_collision =
        ev::apply_bp_reversible_boltzmann_collision_step(
            f,
            nullptr,
            r,
            w,
            L,
            geometry.radial_velocity.width,
            geometry.legacy_Lq,
            0.5 * dt,
            p,
            mpi);
    const ev::KineticInvariantLedger final =
        ev::evaluate_kinetic_invariant_ledger(f, geometry, p);
    assert(ev::kinetic_relative_error(
        after_transport.mass, final.mass) < 5.0e-13);
    assert(ev::kinetic_relative_error(
        after_transport.kinetic_energy, final.kinetic_energy) < 5.0e-13);
    std::cout << "Validated Strang step: escaped="
              << transport.diagnostics.escaped_mass
              << " midpoint_density_difference="
              << midpoint_density_relative_difference
              << " midpoint_limited_faces="
              << transport.diagnostics.midpoint_predictor_limited_faces
              << " fallback_fraction="
              << ((transport.diagnostics.source_mass > 0.0)
                      ? transport.diagnostics.quadrature_fallback_mass
                            / transport.diagnostics.source_mass
                      : 0.0)
              << " collision_omission="
              << std::max(
                    first_collision.max_shell_remap_skipped_fraction,
                    second_collision.max_shell_remap_skipped_fraction)
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--configuration-only") {
        test_collision_step_info_merge();
        test_cell_average_configuration_contracts();
        std::cout << "Configuration and metadata tests passed\n";
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--composite-eta-only") {
        test_composite_eta_axis();
        std::cout << "Composite eta-grid tests passed\n";
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--restart-only") {
        test_restart_representation_guard();
        std::cout << "Restart representation tests passed\n";
        return 0;
    }
    if (argc == 2
        && std::string(argv[1]) == "--hamiltonian-reconstruction-only") {
        test_hamiltonian_quadratic_reconstruction();
        std::cout << "Hamiltonian reconstruction tests passed\n";
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--polygon-only") {
        test_extended_polygon_partition();
        std::cout << "Polygon-moment tests passed\n";
        return 0;
    }
    if ((argc == 2 || argc == 3)
        && std::string(argv[1]) == "--hamiltonian-transport-only") {
        const int eta_quadrature_points = argc == 3
            ? std::stoi(argv[2])
            : 2;
        test_hamiltonian_sldg_reconstruction_and_field_timestep(
            eta_quadrature_points);
        std::cout << "Hamiltonian transport tests passed\n";
        return 0;
    }
    if (argc == 2
        && std::string(argv[1]) == "--hamiltonian-eta-convergence-only") {
        test_hamiltonian_eta_quadrature_convergence();
        std::cout << "Hamiltonian eta-convergence tests passed\n";
        return 0;
    }
    if (argc == 2
        && std::string(argv[1]) == "--delayed-inner-carrier-only") {
        test_delayed_inner_carrier_orbits_and_restart();
        std::cout << "Delayed inner-carrier tests passed\n";
        return 0;
    }
    if (argc == 2
        && std::string(argv[1]) == "--streamed-collision-only") {
        test_streamed_reversible_boltzmann();
        std::cout << "Streamed collision tests passed\n";
        return 0;
    }
    test_parallel_exception_propagation();
    test_collision_step_info_merge();
    test_mass_weighted_collision_clock();
    test_canonical_row_roundoff_positivity_repair();
    test_high_order_cell_quadrature();
    test_composite_eta_axis();
    test_cell_average_configuration_contracts();
    test_reversible_omission_budget_guard();
    test_phase_space_geometry_and_ledger();
    test_restart_representation_guard();
    test_analytic_cell_average_initialization();
    test_specular_wall_reconstruction();
    test_reflection_aware_sldg_refinement();
    test_directional_sldg_refinement_choice();
    test_extended_polygon_partition();
    test_reversible_projection_and_maxwellian();
    test_sampled_reversible_boltzmann();
    test_streamed_reversible_boltzmann();
    test_implicit_fractional_reverse_reaction();
    test_conservative_sldg_identity_and_equilibrium();
    test_hamiltonian_sldg_reconstruction_and_field_timestep();
    test_hamiltonian_eta_quadrature_convergence();
    test_reflecting_wall_orbit_regularization();
    test_delayed_inner_carrier_orbits_and_restart();
    test_validated_strang_step();
    std::cout << "Fully kinetic module tests passed\n";
    return 0;
}
