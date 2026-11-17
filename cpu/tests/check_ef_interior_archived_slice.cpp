// Optional integration check of the exact archived EF marker state. This
// avoids regenerating a Linux IC bit-for-bit on another floating-point ABI;
// it does not weaken any production IC/manifest checks. The binary reader
// still checks the archive's internal fingerprint. Pass a SHA-256-verified
// null-data file, an unused output directory, and an end advanced time.
#include <cassert>
#include "../src/drivers/einstein_vlasov_wpic.cpp"

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    namespace ef = ev::eddington_finkelstein;
    const bool synthetic = std::string(argv[1]) == "--synthetic";
    EFNullInitialData initial;
    if (synthetic) {
        // Coupled regression: a small infalling shell around a pre-existing
        // black hole. This is a test fixture, not the user's collapse IC.
        initial.conserved_adm_mass = 1.0;
        for (int i = 0; i < 256; ++i) {
            ef::Marker marker;
            marker.r = 2.01 + 0.2 * (i + 0.5) / 256;
            marker.p_r = 1.0 / (1.0 + std::sqrt(2.0 / marker.r));
            marker.eta = 0.02;
            marker.rest_weight = 1e-5;
            marker.id = marker.source_id = i;
            initial.markers.push_back(marker);
        }
    } else {
        initial = read_null_initial_data(argv[1]);
    }
    const std::filesystem::path out(argv[2]);
    if (std::filesystem::exists(out)) throw std::runtime_error("Output already exists");
    std::filesystem::create_directories(out);
    const double end = std::stod(argv[3]);
    auto p = base_parameters(synthetic ? 512 : 2048, 0.005, synthetic ? 8.0 : 48.0);
    p.inner_boundary = "origin_parity";
    p.radial_grid = "linear";
    const auto grid = ef::make_field_grid(p, p.Nr);
    auto markers = initial.markers;
    ef::EvolutionOptions controls;
    controls.threads = synthetic ? 1 : 4;
    controls.characteristic_substeps = 2;
    controls.fixed_adm_mass = initial.conserved_adm_mass;
    ef::ConstraintOptions constraints;
    constraints.fixed_adm_mass = initial.conserved_adm_mass;
    auto field = ef::deposit_and_solve_field(markers, grid,
        ev::WeightedPICParticleShape::tsc, controls.threads, constraints);
    std::ofstream history(out / "high_compactness_history.csv");
    history << std::setprecision(17)
        << "step,advanced_time,total_rest_weight,active_rest_weight,excised_rest_weight,"
           "apparent_horizon_radius,excision_radius,active_trapped_rest_weight,active_min_radius\n";
    std::ofstream profiles(out / "high_compactness_profiles.csv");
    profiles << std::setprecision(17)
        << "step,advanced_time,r,a,b,m,compactness,T_rr,S,T_vv,T_vr,T_theta_theta,"
           "rest_weight_density,metric_sample_kind,metric_cell,face_left_r,face_right_r,"
           "face_left_a,face_right_a,face_left_b,face_right_b,rho_infall,excision_mask\n";
    bool active_inside = false, density_inside = false, deep_reached = false;
    double v = 0.0;
    int step = 0;
    auto record = [&]() {
        long double trapped = 0;
        double rmin = std::numeric_limits<double>::infinity();
        for (const auto& marker : markers) {
            if (marker.state != ef::MarkerState::active) continue;
            rmin = std::min(rmin, marker.r);
            if (ef::sample_metric(marker.r, grid, field.metric).a < 0) trapped += marker.rest_weight;
        }
        const double rh = field.metric.apparent_horizon_radius;
        active_inside = active_inside || trapped > 0;
        deep_reached = deep_reached || (rh > 0 && rmin < 0.2 * rh);
        assert(field.metric.excision_radius <= 0.1 * rh + 1e-12);
        const auto stress = deposit_ef_covariant_stress_audit(markers, grid, field.metric);
        for (std::size_t i = 0; i < grid.radius.size(); ++i) {
            assert(std::isfinite(stress.rho_infall[i]) && stress.rho_infall[i] >= 0);
            density_inside = density_inside || (field.metric.a[i] < 0
                && grid.radius[i] > field.metric.excision_radius && stress.rho_infall[i] > 0);
        }
        history << step << ',' << v << ',' << field.total_rest_weight << ','
            << field.sources.active_rest_weight << ',' << field.sources.excised_rest_weight << ','
            << rh << ',' << field.metric.excision_radius << ',' << static_cast<double>(trapped)
            << ',' << (std::isfinite(rmin) ? rmin : -1.0) << '\n';
        write_ef_high_profile(profiles, step, v, markers, grid, field);
        history.flush(); profiles.flush();
        std::cout << "step=" << step << " v=" << v << " rmin=" << rmin
            << " rex=" << field.metric.excision_radius << " active="
            << field.sources.active_rest_weight << " trapped=" << static_cast<double>(trapped) << std::endl;
    };
    record();
    while (v < end) {
        double dt = std::min(ef::choose_timestep(markers, grid, field.metric, 0.2, 0.004), end - v);
        auto result = ef::advance_self_consistent(markers, dt, grid, controls);
        field = std::move(result.endpoint_field);
        v += dt; ++step;
        if (step % (synthetic ? 10 : 50) == 0 || v >= end) record();
    }
    assert(active_inside && density_inside && deep_reached);
    std::cout << (synthetic ? "Synthetic coupled" : "Archived")
        << " EF interior integration check passed" << std::endl;
}
