#include "ev/relativistic_boltzmann/coupled_evolution.hpp"
#include "ev/relativistic_boltzmann/ef_frame.hpp"
#include <iomanip>
#include <iostream>
#include <tuple>

static void check(bool ok, const char* text) {
    if (!ok) throw std::runtime_error(text);
}

static void near(double a, double b, double tolerance, const char* text) {
    check(std::abs(a-b) <= tolerance*std::max({std::abs(a),std::abs(b),1.0e-100}),text);
}

static void reject(const std::function<void()>& action, const char* text) {
    bool failed = false;
    try { action(); } catch (const std::runtime_error&) { failed = true; }
    check(failed,text);
}

static void frame_and_rate_tests() {
    eb::Model m;
    m.material_mass = 0.3;
    m.cross_section = 0.01;
    eb::Cell initial;
    initial.next_id = 3;
    initial.packets = {{0.3,2,0.2,0,1,1},{0.9,2,2,0,2,2}};
    const auto ka = eb::lift(initial.packets[0],m,eb::Slice::advanced_null,0);
    const auto kb = eb::lift(initial.packets[1],m,eb::Slice::advanced_null,0);
    for (double a : {0.8,0.0,-0.5}) {
        eb::EFPoint point{a,0.7,2};
        const auto g = eb::ef_frame(point);
        for (const auto& p : initial.packets) {
            const auto k = eb::lift(p,m,eb::Slice::advanced_null,0);
            const auto coordinate = rba::from_tetrad(g,k);
            const auto covariant = rba::multiply(g.metric,coordinate);
            near(coordinate[0],m.material_mass*p.radial/point.b,1.0e-13,"EF k^v mapping failed.");
            near(covariant[1],m.material_mass*p.radial,1.0e-13,"EF physical k_r mapping failed.");
            near(covariant[0],eb::ef_covariant_energy(p,m,point),1.0e-13,"EF covariant energy failed.");
            near(rba::dot(g.metric,coordinate,coordinate),-m.material_mass*m.material_mass,
                 2.0e-13,"Horizon-regular physical mass shell failed.");
            near(k.t/m.material_mass,ev::eddington_finkelstein::infall_observer_energy(p.r,p.radial,p.eta),
                 1.0e-14,"Maintained EF observer convention mismatch.");
            auto projected = p;
            eb::project(projected,k,m,eb::Slice::advanced_null);
            near(projected.radial,p.radial,1.0e-14,"Positive canonical projection failed.");
        }
    }
    eb::Geometry g;
    g.slice = eb::Slice::advanced_null;
    g.time_factor = 0.7;
    const double factor = eb::pair_factor(initial.packets[0],initial.packets[1],ka,kb,m,g);
    near(factor,12.375,1.0e-13,"Independent null flux factor failed.");
    check(factor > 2,"Null flux fixture did not exceed the spacelike majorant.");
    const double qmax = 3;
    const double probability = -std::expm1(-qmax*g.time_factor*m.cross_section*factor);
    const int trials = 5000;
    rb::Random seeds{9131};
    int first = 0;
    for (int i = 0; i < trials; ++i) {
        auto state = initial;
        state.random.state = seeds.next();
        eb::advance_cell(state,1,m,g);
        first += state.collisions > 0;
        const auto before = eb::ef_stress(initial,m,0.8,0.7,1);
        const auto after = eb::ef_stress(state,m,0.8,0.7,1);
        near(before.rest_crossing,after.rest_crossing,4.0e-13,"Null rest weight changed.");
        near(before.t_rr,after.t_rr,4.0e-13,"Co-located null T_rr changed.");
        near(before.s,after.s,4.0e-13,"Co-located null S changed.");
        near(before.t_vr,after.t_vr,4.0e-13,"Co-located covariant energy changed.");
    }
    const double observed = static_cast<double>(first)/trials;
    const double error = std::sqrt(probability*(1-probability)/trials);
    check(std::abs(observed-probability) < 5*error,"EF absolute first-event rate failed.");
    auto state = initial;
    eb::advance_cell(state,0.37,m,g);
    const auto bytes = eb::serialize_cell(state,m,g);
    auto restarted = eb::deserialize_cell(bytes,m,g);
    eb::advance_cell(state,1,m,g);
    eb::advance_cell(restarted,1,m,g);
    check(eb::serialize_cell(state,m,g) == eb::serialize_cell(restarted,m,g),"EF model/RNG/optical restart failed.");
    auto wrong = m;
    wrong.material_mass *= 2;
    reject([&] { eb::deserialize_cell(bytes,wrong,g); },"Changed material mass restart accepted.");
    auto wrong_slice = g;
    wrong_slice.slice = eb::Slice::schwarzschild;
    reject([&] { eb::deserialize_cell(bytes,m,wrong_slice); },"Changed slice restart accepted.");
    auto separated = initial;
    separated.packets[1].r = 2.1;
    reject([&] { eb::advance_cell(separated,1,m,g); },"Null-separated two-way interaction accepted as point local.");
    std::cout << "EF null_flux=" << factor << " first_event=" << observed
              << " expected=" << probability << " sigma=" << error << '\n';
}

static ev::Params parameters() {
    ev::Params p;
    p.physics_model = "einstein_vlasov";
    p.vp_universe = "static";
    p.Nr = 16;
    p.r_min = 0;
    p.r_max = 5;
    p.radial_grid = "linear";
    p.kinetic_grid = "cell_average_eta";
    p.inner_boundary = "origin_parity";
    p.outer_boundary = "outflow_vacuum";
    p.bp_collision_operator = "none";
    return p;
}

static bool equal_marker(const ev::WeightedPICMarker& a, const ev::WeightedPICMarker& b) {
    return std::tie(a.mass,a.r,a.w,a.eta,a.inner_boundary_speed,a.remaining_inner_time,
                    a.internal_energy,a.internal_velocity_dispersion,a.id,a.source_cell,a.state)
        == std::tie(b.mass,b.r,b.w,b.eta,b.inner_boundary_speed,b.remaining_inner_time,
                    b.internal_energy,b.internal_velocity_dispersion,b.id,b.source_cell,b.state);
}

static bool equal_marker(const ev::eddington_finkelstein::Marker& a,
                         const ev::eddington_finkelstein::Marker& b) {
    return std::tie(a.rest_weight,a.r,a.p_r,a.eta,a.id,a.source_id,a.state)
        == std::tie(b.rest_weight,b.r,b.p_r,b.eta,b.id,b.source_id,b.state);
}

static void coupled_tests() {
    const auto p = parameters();
    const auto scgrid = ev::make_weighted_einstein_vlasov_pic_field_grid(p,16);
    const auto efgrid = ev::eddington_finkelstein::make_field_grid(p,16);
    eb::SCState sc;
    sc.collision_clocks.resize(16);
    for (std::uint64_t i = 1; i <= 32; ++i) {
        ev::WeightedPICMarker marker;
        marker.mass = (1+i%3)*1.0e-6;
        marker.r = 2;
        marker.w = (i%2 ? -1 : 1)*0.7;
        marker.eta = 0.01;
        marker.id = i;
        marker.source_cell = 5;
        sc.markers.push_back(marker);
    }
    sc.next_id = 33;
    const auto field = ev::deposit_and_solve_weighted_einstein_vlasov_pic_field(sc.markers,scgrid,
        ev::WeightedPICParticleShape::tsc,1);
    eb::EFState ef;
    ef.collision_clocks.resize(16);
    ef.next_id = sc.next_id;
    for (const auto& marker : sc.markers) {
        // Mechanical map used only for a small code-control fixture. A general
        // SC-to-null initial-data import also needs its physical slice mapping.
        const auto metric = ev::weighted_einstein_vlasov_pic_detail::sample_metric_on_grid(marker.r,scgrid,field.metric);
        ef.markers.push_back(ev::eddington_finkelstein::from_schwarzschild_marker(marker,std::exp(-2*metric.lambda)));
    }
    ev::WeightedEinsteinVlasovPICOptions scopt;
    scopt.threads = 1;
    scopt.require_vacuum_outer_boundary = true;
    ev::eddington_finkelstein::EvolutionOptions efopt;
    efopt.threads = 1;
    efopt.require_vacuum_outer_boundary = true;
    efopt.enable_excision = false;
    eb::SplitControls c;
    auto sc_control = sc.markers;
    const auto original_sc = ev::advance_weighted_einstein_vlasov_pic_self_consistent(sc_control,1.0e-4,scgrid,scopt);
    auto sc_off = sc;
    const auto off_sc = eb::advance_sc(sc_off,1.0e-4,scgrid,scopt,c);
    for (std::size_t i = 0; i < sc_control.size(); ++i)
        check(equal_marker(sc_control[i],sc_off.markers[i]),"SC sigma0 physical marker differs from EV.");
    check(off_sc.field.metric.mu == original_sc.endpoint_field.metric.mu
          && off_sc.field.metric.m == original_sc.endpoint_field.metric.m
          && off_sc.field.metric.lambda == original_sc.endpoint_field.metric.lambda,
          "SC sigma0 metric differs from EV.");
    auto ef_control = ef.markers;
    const auto original_ef = ev::eddington_finkelstein::advance_self_consistent(ef_control,1.0e-4,efgrid,efopt);
    auto ef_off = ef;
    const auto off_ef = eb::advance_ef(ef_off,1.0e-4,efgrid,efopt,c);
    for (std::size_t i = 0; i < ef_control.size(); ++i)
        check(equal_marker(ef_control[i],ef_off.markers[i]),"EF sigma0 physical marker differs from EV.");
    check(off_ef.field.metric.a == original_ef.endpoint_field.metric.a
          && off_ef.field.metric.b == original_ef.endpoint_field.metric.b
          && off_ef.field.metric.m == original_ef.endpoint_field.metric.m,
          "EF sigma0 metric differs from EV.");
    // Large test-only area ensures many events in the tiny coupled interval.
    c.model.cross_section = 1.0e10;
    c.allow_radial_cell_approximation = true;
    const auto on_sc = eb::advance_sc(sc,1.0e-4,scgrid,scopt,c);
    const auto on_ef = eb::advance_ef(ef,1.0e-4,efgrid,efopt,c);
    check(on_sc.collisions > 0 && on_ef.collisions > 0,"Actual coupled SC/EF collisions were not exercised.");
    check(on_sc.nonlocal_collisions > 0 && on_ef.nonlocal_collisions > 0,
          "Finite-cell diagnostics missed separated transported packets.");
    near(on_sc.streaming.rest_mass_before,on_sc.field.rest_mass,1.0e-12,"Coupled SC represented mass changed.");
    near(on_ef.streaming.rest_weight_before,on_ef.field.total_rest_weight,1.0e-12,"Coupled EF represented mass changed.");
    std::cout << "coupled SC_events=" << on_sc.collisions << " EF_events=" << on_ef.collisions
              << " SC_delta_ADM=" << on_sc.collision_adm_change << " EF_delta_ADM=" << on_ef.collision_adm_change << '\n';
}

int main() {
    try {
        std::cout << std::setprecision(17);
        frame_and_rate_tests();
        coupled_tests();
        std::cout << "PASS coupled SC/EF suite\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
