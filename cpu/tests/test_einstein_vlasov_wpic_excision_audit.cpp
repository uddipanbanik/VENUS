// Exercise the production driver's accepted-state excision audit directly.
// Including its implementation keeps this regression on the actual audit
// and persistence code used by the task-specific executable.
#include "../src/drivers/einstein_vlasov_wpic.cpp"

#include <cassert>

namespace excision_test_ef = ev::eddington_finkelstein;

int main() {
    {
        const std::vector<ev::WeightedPICMarker> initial{
            make_marker(.2, 2.0, -.3, .04, 11),
            make_marker(.3, 3.0, -.4, .06, 12),
        };
        auto transported = initial;
        std::reverse(transported.begin(), transported.end());
        transported[0].r = 1.2;
        transported[0].w = -.8;
        validate_restart_initial_marker_invariants(transported, initial);
        std::vector<excision_test_ef::Marker> null_markers;
        for (const auto& marker : transported) {
            excision_test_ef::Marker mapped;
            mapped.rest_weight = marker.mass;
            mapped.eta = marker.eta;
            mapped.id = marker.id;
            mapped.source_id = marker.source_cell;
            mapped.r = .9;
            mapped.p_r = 1.2;
            null_markers.push_back(mapped);
        }
        validate_restart_initial_marker_invariants(null_markers, initial);
        for (int defect = 0; defect < 5; ++defect) {
            auto wrong = null_markers;
            if (defect == 0) wrong[0].rest_weight *= 1.01;
            if (defect == 1) wrong[0].eta *= 2.0;
            if (defect == 2) ++wrong[0].source_id;
            if (defect == 3) wrong[0].id = 99;
            if (defect == 4) wrong[0] = wrong[1];
            bool rejected = false;
            try { validate_restart_initial_marker_invariants(wrong, initial); }
            catch (const std::runtime_error&) { rejected = true; }
            assert(rejected);
        }
    }
    ev::Params p;
    p.physics_model = "einstein_vlasov";
    p.vp_universe = "static";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "uniform";
    p.Nr = 1024;
    p.Nw = 4;
    p.NL = 2;
    p.r_min = 0.25;
    p.r_max = 8.0;
    p.radial_grid = "linear";
    p.inner_boundary = "origin_parity";
    p.outer_boundary = "outflow_vacuum";
    const auto grid = excision_test_ef::make_field_grid(p, p.Nr);
    {
        auto reflected = p;
        reflected.outer_boundary = "reflecting";
        bool rejected = false;
        try { (void)excision_test_ef::make_field_grid(reflected, reflected.Nr); }
        catch (const std::runtime_error&) { rejected = true; }
        std::cout << "EF reflecting outer configuration rejected="
                  << rejected << std::endl;
        assert(rejected);
    }
    excision_test_ef::Marker shell;
    shell.rest_weight = 2.0;
    shell.r = 1.0;
    shell.p_r = 1.0;
    shell.id = 1;
    const std::vector<excision_test_ef::Marker> markers{shell};
    auto field = excision_test_ef::deposit_and_solve_field(
        markers, grid, ev::WeightedPICParticleShape::tsc, 1);
    assert(field.metric.apparent_horizon_found);
    assert(field.metric.excision_radius == 0.0);
    // Reproduce a stale retained boundary in the regular central cavity.
    // It is inside the outer horizon, but outgoing null rays have dr/dv>0.
    const std::size_t cavity = 3;
    field.metric.excision_radius = grid.radius[cavity];
    assert(field.metric.excision_radius < field.metric.apparent_horizon_radius);
    assert(field.metric.a[cavity] > 0.0 && field.metric.b[cavity] > 0.0);
    EFPostHorizonAuditState audit;
    update_ef_post_horizon_audit(
        audit, markers, field, field.metric.adm_mass, true);
    std::cout << std::setprecision(17)
              << "annulus audit: r_ex=" << field.metric.excision_radius
              << " r_AH=" << field.metric.apparent_horizon_radius
              << " a_ex=" << field.metric.a[cavity]
              << " b_ex=" << field.metric.b[cavity]
              << " outgoing_null_speed="
              << 0.5 * field.metric.a[cavity] * field.metric.b[cavity]
              << " causal=" << audit.causal_excision_inside_horizon_seen
              << std::endl;
    assert(!audit.causal_excision_inside_horizon_seen);
    assert(!audit.excised_interior_rest_weight_seen);
    assert(!audit.causal_excision_valid_throughout);

    excision_test_ef::ConstraintOptions constraints;
    constraints.fixed_adm_mass = 1.0;
    excision_test_ef::Marker probe = shell;
    probe.rest_weight = 1.0e-12;
    probe.r = 3.0;
    const std::vector<excision_test_ef::Marker> probes{probe};
    const auto black_hole = excision_test_ef::deposit_and_solve_field(
        probes, grid, ev::WeightedPICParticleShape::tsc, 1, constraints);
    const double safe_radius = black_hole.metric.excision_radius;
    assert(safe_radius > 0.0);
    assert(excision_test_ef::causal_excision_boundary(
        black_hole.metric, safe_radius, -0.01, 4));
    const std::size_t boundary = static_cast<std::size_t>(std::lower_bound(
        grid.radius.begin(), grid.radius.end(), safe_radius) - grid.radius.begin());
    // Every interpolation knot matters. Negative center values alone can
    // conceal an untrapped face or an invalid lapse in the outward buffer.
    for (int defect = 0; defect < 3; ++defect) {
        auto corrupt = black_hole.metric;
        if (defect == 0) corrupt.a[boundary + 2] = 0.1;
        if (defect == 1) corrupt.face_a[boundary + 2] = 0.1;
        if (defect == 2) corrupt.face_b[boundary + 2] = 0.0;
        assert(!excision_test_ef::causal_excision_boundary(
            corrupt, safe_radius, -0.01, 4));
        // This models the former max(recomputed,applied) endpoint override.
        corrupt.excision_radius = 0.0;
        bool rejected = false;
        try {
            excision_test_ef::retain_causal_excision_boundary(
                corrupt, safe_radius, constraints);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        assert(rejected && corrupt.excision_radius == 0.0);
    }
    auto retained = black_hole.metric;
    retained.excision_radius = 0.0;
    excision_test_ef::retain_causal_excision_boundary(
        retained, safe_radius, constraints);
    assert(retained.excision_radius == safe_radius);
    {
        auto tighter = constraints;
        tighter.excision_horizon_fraction = 0.01;
        auto candidate = black_hole.metric;
        candidate.excision_radius = 0.0;
        bool rejected = false;
        try { excision_test_ef::retain_causal_excision_boundary(
            candidate, safe_radius, tighter); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && candidate.excision_radius == 0.0);
    }

    EFPostHorizonAuditState good;
    update_ef_post_horizon_audit(good, probes, black_hole, 1.0, true);
    assert(good.complete && good.causal_excision_valid_throughout);
    assert(good.causal_excision_inside_horizon_seen);
    assert(good.accepted_step_samples == 1);
    auto interrupted = good;
    update_ef_post_horizon_audit(
        interrupted, markers, field, field.metric.adm_mass, true);
    update_ef_post_horizon_audit(interrupted, probes, black_hole, 1.0, true);
    assert(interrupted.causal_excision_inside_horizon_seen);
    assert(!interrupted.causal_excision_valid_throughout);
    assert(interrupted.accepted_step_samples == 3);

    // Version 2 persists the causal check. Version-1 radius-only evidence
    // remains readable for evolution while its certification is incomplete.
    const auto directory = std::filesystem::temp_directory_path()
        / ("venus_ef_excision_audit_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    {
        RunOptions options;
        options.output_dir = directory / "low_initial_density";
        std::filesystem::create_directories(options.output_dir);
        options.threads = 1;
        options.low_markers = 256;
        options.low_radial_cells = 32;
        options.low_total_mass = .02;
        options.low_velocity_factor = 1.0;
        options.low_t_end = .001;
        options.low_dt_max = .001;
        options.low_output_steps = 1;
        (void)run_low_compactness(options);
        const auto rows = read_numeric_csv_history(
            options.output_dir / "low_compactness_history.csv", 19);
        const auto& initial = rows.front();
        // Canonicalization increases each EV invariant rest weight. Since
        // TSC weights sum to one, the initial positive shell-mass difference
        // equals the total-weight difference divided by the mean total.
        const double expected = (initial[3] - initial[11])
            / (.5 * (initial[3] + initial[11]));
        assert(expected > 1.0e-4);
        assert(std::abs(initial[16] - expected) < 2.0e-15);
        assert(initial[17] == 0.0 && initial[18] == 0.0);
    }
    const auto path = directory / "audit.state";
    write_ef_post_horizon_audit(path, good, 1.0, 1);
    const auto loaded = read_ef_post_horizon_audit(path, 1.0, 1);
    assert(loaded.complete && loaded.causal_excision_valid_throughout);
    assert(loaded.causal_excision_inside_horizon_seen);
    std::ifstream persisted(path);
    std::string header, payload;
    std::getline(persisted, header);
    std::getline(persisted, payload);
    persisted.close();
    assert(header == "VENUS_EF_POST_HORIZON_AUDIT 2");
    {
        std::ofstream legacy(path);
        legacy << "VENUS_EF_POST_HORIZON_AUDIT 1\n"
               << payload.substr(0, payload.find_last_of(' ')) << '\n';
    }
    const auto old = read_ef_post_horizon_audit(path, 1.0, 1);
    assert(!old.complete && !old.causal_excision_valid_throughout);
    assert(!old.causal_excision_inside_horizon_seen);
    assert(!old.excised_interior_rest_weight_seen);
    const auto restart_path = directory / "restart.bin";
    EFRestartState interior{probes, 1.0, 1, 1.0, false};
    write_ef_restart(restart_path, interior);
    const auto loaded_restart = read_ef_restart(restart_path);
    assert(!loaded_restart.enable_excision);
    assert(loaded_restart.markers[0].r == probe.r);
    // Altering the authenticated mode byte without updating its fingerprint
    // must fail even when the active marker records themselves are valid.
    {
        std::fstream tampered(restart_path,
            std::ios::binary | std::ios::in | std::ios::out);
        tampered.seekp(44);
        const char enabled = 1;
        tampered.write(&enabled, 1);
    }
    bool rejected_mode_tampering = false;
    try { (void)read_ef_restart(restart_path); }
    catch (const std::runtime_error&) { rejected_mode_tampering = true; }
    assert(rejected_mode_tampering);
    interior.markers[0].state = excision_test_ef::MarkerState::excised;
    bool rejected_missing_particles = false;
    try { write_ef_restart(restart_path, interior); }
    catch (const std::runtime_error&) { rejected_missing_particles = true; }
    assert(rejected_missing_particles);
    // An original v1 restart has no mode byte and denotes enabled excision.
    EFRestartState legacy_restart{probes, 1.0, 1, 1.0};
    {
        std::ofstream legacy_file(restart_path, std::ios::binary);
        const char magic[16] = {'V','E','N','U','S','E','V','E','F','R','S','T','1',0,0,0};
        legacy_file.write(magic, 16);
        write_physical_ic_scalar<std::uint32_t>(legacy_file, 1);
        write_physical_ic_scalar(legacy_file, legacy_restart.advanced_time);
        write_physical_ic_scalar(legacy_file, legacy_restart.step);
        write_physical_ic_scalar(legacy_file, legacy_restart.conserved_adm_mass);
        write_physical_ic_scalar<std::uint64_t>(legacy_file, probes.size());
        write_physical_ic_scalar(legacy_file, ef_restart_fingerprint(legacy_restart, false));
        for (const auto& value : probes) {
            write_physical_ic_scalar(legacy_file, value.rest_weight);
            write_physical_ic_scalar(legacy_file, value.r);
            write_physical_ic_scalar(legacy_file, value.p_r);
            write_physical_ic_scalar(legacy_file, value.eta);
            write_physical_ic_scalar(legacy_file, value.id);
            write_physical_ic_scalar(legacy_file, value.source_id);
            write_physical_ic_scalar(legacy_file, static_cast<std::uint8_t>(value.state));
        }
    }
    const auto original = read_ef_restart(restart_path);
    assert(original.enable_excision);
    assert(original.markers[0].r == probe.r);
    std::filesystem::remove_all(directory);

    const auto flat = excision_test_ef::deposit_and_solve_field(
        {}, grid, ev::WeightedPICParticleShape::tsc, 1);
    {
        // Move the isolated outer boundary from 8 to 16 at fixed dr, with
        // identical interior matter and fixed ADM mass. The added region is
        // vacuum, so b(infinity)=1 and the interior field/orbits agree.
        auto wider = p;
        wider.Nr *= 2;
        wider.r_max *= 2.0;
        const auto wide_grid = excision_test_ef::make_field_grid(wider, wider.Nr);
        auto matter = probe;
        matter.rest_weight = 0.01;
        matter.p_r = 0.8;
        matter.eta = 0.03;
        const std::vector<excision_test_ef::Marker> cloud{matter};
        const auto narrow_field = excision_test_ef::deposit_and_solve_field(
            cloud, grid, ev::WeightedPICParticleShape::tsc, 1, constraints);
        const auto wide_field = excision_test_ef::deposit_and_solve_field(
            cloud, wide_grid, ev::WeightedPICParticleShape::tsc, 1, constraints);
        double a_error = 0.0, b_error = 0.0;
        for (std::size_t i = 0; i < grid.radius.size(); ++i) {
            assert(grid.radius[i] == wide_grid.radius[i]);
            a_error = std::max(a_error,
                std::abs(narrow_field.metric.a[i] - wide_field.metric.a[i]));
            b_error = std::max(b_error,
                std::abs(narrow_field.metric.b[i] - wide_field.metric.b[i]));
        }
        auto narrow_orbit = cloud, wide_orbit = cloud;
        excision_test_ef::advance_frozen_metric(
            narrow_orbit, 0.1, grid, narrow_field.metric, 2);
        excision_test_ef::advance_frozen_metric(
            wide_orbit, 0.1, wide_grid, wide_field.metric, 2);
        const double r_error = std::abs(narrow_orbit[0].r - wide_orbit[0].r);
        const double p_error = std::abs(narrow_orbit[0].p_r - wide_orbit[0].p_r);
        assert(a_error <= 4.0 * std::numeric_limits<double>::epsilon());
        assert(b_error <= 4.0 * std::numeric_limits<double>::epsilon());
        assert(r_error <= 4.0 * std::numeric_limits<double>::epsilon());
        assert(p_error <= 4.0 * std::numeric_limits<double>::epsilon());
        std::cout << "EF outer-domain 8->16 at fixed dr: max_da=" << a_error
                  << " max_db=" << b_error << " dr_orbit=" << r_error
                  << " dp_orbit=" << p_error << '\n';
    }
    {
        auto outgoing = probe;
        outgoing.r = 7.99;
        outgoing.p_r = 2.0;
        std::vector<excision_test_ef::Marker> cloud{outgoing};
        excision_test_ef::advance_frozen_metric(cloud, 0.1, grid, flat.metric);
        assert(cloud.front().state == excision_test_ef::MarkerState::escaped);
        assert(std::abs(cloud.front().r - 8.0275) < 2.0e-14);
        const auto deposited = excision_test_ef::deposit_sources(
            cloud, grid.mesh, ev::WeightedPICParticleShape::tsc);
        assert(deposited.escaped_rest_weight == outgoing.rest_weight);
        assert(deposited.active_rest_weight == 0.0);
        assert(deposited.deposited_rest_weight == 0.0);
    }
    {
        auto incoming = probe;
        incoming.r = 0.01;
        incoming.p_r = 0.5;
        std::vector<excision_test_ef::Marker> cloud{incoming};
        excision_test_ef::EvolutionOptions evolution;
        evolution.threads = 1;
        bool rejected = false;
        try { (void)excision_test_ef::advance_self_consistent(
            cloud, 0.02, grid, evolution); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
        assert(cloud.front().r == incoming.r);
        assert(cloud.front().p_r == incoming.p_r);
        assert(cloud.front().state == incoming.state);
    }
    {
        // True-origin parity permits a current at the first positive sample.
        // Zero-duration Eulerian advection must preserve that regular state.
        ev::Params point = p;
        point.Nr = 3; point.Nw = 3; point.NL = 2;
        point.r_min = 0.25; point.r_max = 1.0;
        point.integrator = "semi_lagrangian_kick_drift_kick";
        const std::vector<double> r{0.25, 0.5, 1.0};
        const std::vector<double> w{-1.0, 0.0, 1.0};
        const std::vector<double> ell{0.0, 0.1};
        ev::Metric metric;
        metric.m.assign(3, 0.0); metric.lambda.assign(3, 0.0);
        metric.mu.assign(3, 0.0); metric.lambda_r.assign(3, 0.0);
        metric.mu_r.assign(3, 0.0); metric.lambda_t.assign(3, 0.0);
        metric.gap.assign(3, 1.0);
        std::vector<double> f(18);
        for (int i = 0; i < point.Nr; ++i)
            for (int j = 0; j < point.Nw; ++j)
                for (int k = 0; k < point.NL; ++k)
                    f[ev::idx(i, j, k, point)] = std::exp(
                        -r[i] * r[i] - w[j] * w[j]
                        -std::pow(ell[k] / r[i], 2) + 0.5 * r[i] * w[j]);
        const auto unchanged = ev::advect(f, r, r, w, ell, metric, 0.0, point);
        assert(unchanged == f);
        assert(f[ev::idx(0, 2, 0, point)] > f[ev::idx(0, 0, 0, point)]);
    }
    {
        // A Schwarzschild isolated exterior also requires matter to remain
        // inside the vacuum boundary, independently of the EF formulation.
        const auto directory = std::filesystem::temp_directory_path()
            / ("venus-sc-outer-domain-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(directory);
        auto small_parameters = p;
        small_parameters.Nr = 64;
        const auto small_grid = ev::make_weighted_einstein_vlasov_pic_field_grid(
            small_parameters, small_parameters.Nr);
        ev::WeightedPICMarker outward;
        outward.mass = 1.0e-3;
        outward.r = 7.99;
        outward.w = 1.0;
        const std::vector<ev::WeightedPICMarker> cloud{outward};
        const auto initial = ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(
            cloud, small_grid, ev::WeightedPICParticleShape::tsc, 1);
        write_wpic_restart(directory / "high_einstein_vlasov.restart", cloud, 0.0, 0);
        const auto lapse = lapse_diagnostics(initial.metric);
        {
            std::ofstream history(directory / "high_compactness_history.csv");
            history << "step,time,dt,rest_mass,adm_mass,max_compactness,min_gap,"
                "min_lapse,max_log_redshift,r10,r50,r90,virial_ratio,"
                "anisotropic_jeans_relative_l1\n"
                << std::setprecision(17) << "0,0,0," << initial.rest_mass << ','
                << initial.adm_mass << ',' << initial.maximum_compactness << ','
                << initial.metric.min_gap << ',' << lapse.first << ',' << lapse.second
                << ",7.99,7.99,7.99,0,-1\n";
            std::ofstream profiles(directory / "high_compactness_profiles.csv");
            profiles << "step,time,r,rho,compactness,m,a,lambda,mu,lambda_t,"
                "p_radial,p_tangential_sum,j,metric_sample_kind,metric_cell,"
                "face_left_r,face_right_r,face_left_lambda,face_right_lambda,"
                "center_mu_r,face_left_mu,face_right_mu,face_left_mu_r,face_right_mu_r\n"
                << std::setprecision(17);
            write_high_profile(profiles, 0, 0.0, small_grid, initial);
        }
        FixedFamilyBoundDiagnostics bound;
        bound.rest_weight = outward.mass;
        bound.adm_mass = initial.adm_mass;
        bound.binding_energy = bound.rest_weight - bound.adm_mass;
        bound.binding_fraction = bound.binding_energy / bound.rest_weight;
        bound.energy_proxy_minimum = bound.energy_proxy_maximum
            = bound.energy_proxy_rest_weighted_mean = std::sqrt(2.0);
        bound.energy_proxy_rest_weight_fraction_ge_one = 1.0;
        write_fixed_family_bound_diagnostics(directory, bound);
        RunOptions options;
        options.resume = true;
        options.coordinates = "schwarzschild";
        options.high_initial_condition = "shell";
        options.high_markers = 1;
        options.high_radial_cells = 64;
        options.high_r_min = 0.25;
        options.high_r_max = 8.0;
        options.high_inner_boundary = "origin_parity";
        options.high_radial_grid = "linear";
        options.high_t_end = 0.2;
        options.high_dt_max = 0.2;
        options.high_output_steps = 1000;
        options.output_dir = directory;
        options.threads = 1;
        const HighSummary limited = run_high_compactness_schwarzschild(options);
        std::cout << "self-gravitating SC outer-domain result=" << limited.outcome
                  << " steps=" << limited.steps << " escaped=" << limited.escaped_rest_mass
                  << " directory=" << directory << '\n';
        assert(limited.outcome == "sc_outer_boundary_evolution_limit");
        assert(limited.steps == 0 && limited.escaped_rest_mass == 0.0);
        assert(limited.schwarzschild_evolution.outer_domain_limited);
        assert(limited.schwarzschild_evolution.final_time == 0.0);
        assert(limited.schwarzschild_evolution.rejected_trial_dt > 0.0);
        assert(!limited.horizon_formed && !limited.virialized_halo);
        write_summary(options, nullptr, &limited);
        const auto saved = ev::read_weighted_pic_restart(
            directory / "high_einstein_vlasov.restart");
        assert(saved.time == 0.0 && saved.step == 0);
        assert(saved.markers.front().r == outward.r
            && saved.markers.front().w == outward.w
            && saved.markers.front().state == ev::WeightedPICMarkerState::active);
        assert(last_csv_step(directory / "high_compactness_history.csv") == 0);
        assert(last_csv_step(directory / "high_compactness_profiles.csv") == 0);
        std::ifstream status_file(directory / "summary.json");
        const std::string status_text((std::istreambuf_iterator<char>(status_file)), {});
        assert(status_text.find("vacuum_outer_boundary_limit") != std::string::npos);
        std::filesystem::remove_all(directory);
    }
    {
        // Resume a small self-gravitating outward shell close to the outer
        // face. The production fixed-ADM vacuum exterior must stop before
        // removing this shell and leaving its mass at the origin.
        const auto directory = std::filesystem::temp_directory_path()
            / ("venus-ef-outer-domain-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(directory);
        auto small_parameters = p;
        small_parameters.Nr = 64;
        const auto small_grid = excision_test_ef::make_field_grid(
            small_parameters, small_parameters.Nr);
        auto outward = probe;
        outward.rest_weight = 1.0e-3;
        outward.r = 7.99;
        outward.p_r = 2.0;
        outward.eta = 0.0;
        const std::vector<excision_test_ef::Marker> cloud{outward};
        const auto initial = excision_test_ef::deposit_and_solve_field(cloud, small_grid);
        write_ef_restart(directory / "high_eddington_finkelstein.restart",
            {cloud, 0.0, 0, initial.metric.adm_mass, true});
        EFPostHorizonAuditState audit;
        update_ef_post_horizon_audit(audit, cloud, initial,
            initial.metric.adm_mass, true);
        write_ef_post_horizon_audit(ef_post_horizon_audit_path(directory), audit, 0.0, 0);
        {
            std::ofstream controls(directory / "ef_interior_controls.txt");
            controls << std::setprecision(17) << "VENUS_EF_INTERIOR_CONTROLS 2\n"
                     << -0.01 << ' ' << 4 << ' ' << 0.1 << " 1\n";
            std::ofstream history(directory / "high_compactness_history.csv");
            history << "step,advanced_time,dt,total_rest_weight,active_rest_weight,"
                "excised_rest_weight,escaped_rest_weight,adm_mass,max_compactness,"
                "min_a,min_b,apparent_horizon_radius,r10,r50,r90,virial_ratio,"
                "excision_radius,anisotropic_jeans_relative_l1,active_marker_count,"
                "active_trapped_rest_weight,active_min_radius\n"
                << std::setprecision(17) << "0,0,0," << outward.rest_weight << ','
                << outward.rest_weight << ",0,0," << initial.metric.adm_mass << ','
                << initial.metric.maximum_compactness << ',' << initial.metric.minimum_a
                << ',' << initial.metric.b.front() << ",-1,7.99,7.99,7.99,0,0,-1,1,0,7.99\n";
            std::ofstream profiles(directory / "high_compactness_profiles.csv");
            profiles << "step,advanced_time,r,a,b,m,compactness,T_rr,S,T_vv,T_vr,"
                "T_theta_theta,rest_weight_density,metric_sample_kind,metric_cell,"
                "face_left_r,face_right_r,face_left_a,face_right_a,face_left_b,"
                "face_right_b,rho_infall,excision_mask\n" << std::setprecision(17);
            write_ef_high_profile(profiles, 0, 0.0, cloud, small_grid, initial);
        }
        FixedFamilyBoundDiagnostics bound;
        bound.rest_weight = outward.rest_weight;
        bound.adm_mass = initial.metric.adm_mass;
        bound.binding_energy = bound.rest_weight - bound.adm_mass;
        bound.binding_fraction = bound.binding_energy / bound.rest_weight;
        bound.energy_proxy_minimum = bound.energy_proxy_maximum
            = bound.energy_proxy_rest_weighted_mean = 1.25;
        bound.energy_proxy_rest_weight_fraction_ge_one = 1.0;
        write_fixed_family_bound_diagnostics(directory, bound);
        RunOptions options;
        options.resume = true;
        options.coordinates = "eddington_finkelstein";
        options.high_initial_condition = "shell";
        options.high_markers = 1;
        options.high_radial_cells = 64;
        options.high_r_min = 0.25;
        options.high_r_max = 8.0;
        options.high_inner_boundary = "origin_parity";
        options.high_radial_grid = "linear";
        options.high_t_end = 0.2;
        options.high_dt_max = 0.2;
        options.high_output_steps = 1000;
        options.output_dir = directory;
        options.threads = 1;
        const HighSummary limited = run_high_compactness_eddington_finkelstein(options);
        std::cout << "self-gravitating EF outer-domain result=" << limited.outcome
                  << " steps=" << limited.steps << " directory=" << directory << '\n';
        assert(limited.outcome == "ef_outer_boundary_evolution_limit");
        assert(limited.ef_termination_reason == "vacuum_outer_boundary_limit");
        assert(limited.ef_spatial_resolution_assessment == "excision_enabled");
        assert(!limited.ef_origin_mass_residual_available);
        assert(limited.ef_density_peak_half_maximum_cells > 0);
        assert(limited.ef_minimum_active_radius_over_first_cell_width > 0.0);
        assert(limited.steps == 0 && limited.ef_final_advanced_time == 0.0);
        assert(!limited.ef_requested_end_time_reached);
        assert(limited.escaped_rest_mass == 0.0 && limited.rest_mass_relative_error == 0.0);
        write_summary(options, nullptr, &limited);
        const auto saved = read_ef_restart(directory / "high_eddington_finkelstein.restart");
        assert(saved.advanced_time == 0.0 && saved.step == 0);
        assert(saved.markers.front().r == outward.r
            && saved.markers.front().p_r == outward.p_r
            && saved.markers.front().state == excision_test_ef::MarkerState::active);
        assert(last_csv_step(directory / "high_compactness_history.csv") == 0);
        assert(last_csv_step(directory / "high_compactness_profiles.csv") == 0);
        std::ifstream summary_file(directory / "summary.json");
        const std::string summary_text((std::istreambuf_iterator<char>(summary_file)), {});
        assert(summary_text.find("vacuum_outer_boundary_limit") != std::string::npos);
        std::filesystem::remove_all(directory);
    }
    std::cout << "EF causal boundary, buffer, retained endpoint and v2/v1 audit "
                 "persistence checks passed\n";
    std::cout << "EF causal-excision audit tests passed\n";
}
