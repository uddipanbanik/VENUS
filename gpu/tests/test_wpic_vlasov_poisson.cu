#include "venus_gpu/wpic.hpp"
#include "ev/central_drift.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint64_t mix64(std::uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

double uniform(std::uint64_t counter) {
    return static_cast<double>(mix64(counter) >> 11)
        * (1.0 / 9007199254740992.0);
}

venus_gpu::RadialMesh make_mesh(int cells, double inner, double outer) {
    std::vector<double> edge(static_cast<std::size_t>(cells) + 1);
    for (int i = 0; i <= cells; ++i) {
        edge[static_cast<std::size_t>(i)] = inner
            + (outer - inner) * static_cast<double>(i) / cells;
    }
    return venus_gpu::RadialMesh::from_edges(edge);
}

void test_marker_upload_validation() {
    venus_gpu::Marker valid;
    valid.mass = 1.0;
    valid.r = 1.0;
    valid.w = -0.25;
    valid.eta = 0.125;
    valid.id = 7;
    valid.internal_energy = 0.2;
    valid.internal_velocity_dispersion = 0.3;
    valid.inner_boundary_speed = 0.4;

    auto rejected = [](const venus_gpu::Marker& marker) {
        try {
            venus_gpu::WpicState state({marker});
            (void)state;  // Construction itself is the validation operation.
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };

    venus_gpu::Marker marker = valid;
    marker.state = static_cast<venus_gpu::MarkerState>(255);
    assert(rejected(marker));

    marker = valid;
    marker.internal_energy = -1.0;
    assert(rejected(marker));
    marker = valid;
    marker.internal_energy = std::numeric_limits<double>::infinity();
    assert(rejected(marker));

    marker = valid;
    marker.internal_velocity_dispersion = -1.0;
    assert(rejected(marker));
    marker = valid;
    marker.internal_velocity_dispersion =
        std::numeric_limits<double>::quiet_NaN();
    assert(rejected(marker));

    marker = valid;
    marker.inner_boundary_speed = -1.0;
    assert(rejected(marker));
    marker = valid;
    marker.inner_boundary_speed = std::numeric_limits<double>::infinity();
    assert(rejected(marker));

    marker = valid;
    marker.remaining_inner_time = -1.0;
    assert(rejected(marker));
    marker = valid;
    marker.remaining_inner_time = std::numeric_limits<double>::quiet_NaN();
    assert(rejected(marker));
    marker = valid;
    marker.remaining_inner_time = std::numeric_limits<double>::infinity();
    assert(rejected(marker));

    marker = valid;
    marker.state = venus_gpu::MarkerState::delayed_inner;
    marker.remaining_inner_time = 0.0;
    assert(rejected(marker));

    marker.remaining_inner_time = 0.75;
    venus_gpu::WpicState delayed_state({marker});
    const auto roundtrip = delayed_state.download();
    assert(roundtrip.size() == 1);
    assert(roundtrip[0].state == venus_gpu::MarkerState::delayed_inner);
    assert(roundtrip[0].remaining_inner_time == 0.75);
}

void test_mass_conservative_cic_and_poisson() {
    constexpr std::size_t count = 200000;
    const venus_gpu::RadialMesh mesh = make_mesh(128, 0.25, 4.25);
    const double q0 = std::pow(mesh.edge.front(), 3) / 3.0;
    const double q1 = std::pow(mesh.edge.back(), 3) / 3.0;
    std::vector<venus_gpu::Marker> markers(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double q = q0 + (q1 - q0)
            * (static_cast<double>(i) + 0.5) / count;
        markers[i].mass = 1.0 / count;
        markers[i].r = std::cbrt(3.0 * q);
        markers[i].w = 0.1 * (2.0 * uniform(i + 17) - 1.0);
        markers[i].eta = 0.02 * uniform(i + 91);
        markers[i].id = i;
    }

    venus_gpu::WpicState state(markers);
    venus_gpu::VlasovPoissonModule vp(mesh);
    venus_gpu::VlasovPoissonOptions options;
    options.gravity = 1.0;
    const venus_gpu::DepositionDiagnostics diagnostics =
        vp.deposit_and_solve(state, 0.0, options);
    const venus_gpu::FieldHost field = vp.download_field();
    const double deposited = std::accumulate(
        field.shell_mass.begin(), field.shell_mass.end(), 0.0);
    assert(std::abs(deposited - 1.0) < 3.0e-12);
    assert(std::abs(diagnostics.deposited_mesh_mass - 1.0) < 3.0e-12);
    assert(std::abs(field.source_mass_edge.back() - 1.0) < 3.0e-12);
    for (std::size_t i = 1; i < field.source_mass_edge.size(); ++i) {
        assert(field.source_mass_edge[i] >= field.source_mass_edge[i - 1]);
    }
    assert(field.phi.back() < 0.0);
    assert(field.phi_r.back() > 0.0);
    std::cout << "mass_conservation_error=" << deposited - 1.0 << '\n';
}

void test_ballistic_exact_central_drift() {
    constexpr std::size_t count = 8192;
    constexpr double dt = 0.015;
    const venus_gpu::RadialMesh mesh = make_mesh(96, 0.25, 4.25);
    std::vector<venus_gpu::Marker> markers(count);
    std::vector<double> expected_r(count), expected_w(count);
    for (std::size_t i = 0; i < count; ++i) {
        venus_gpu::Marker& marker = markers[i];
        marker.mass = 1.0 / count;
        marker.r = 1.0 + 2.0 * uniform(i + 101);
        marker.w = 0.2 * (2.0 * uniform(i + 303) - 1.0);
        marker.eta = 0.02 + 0.08 * uniform(i + 707);
        marker.id = i;
        const double angular_momentum = std::sqrt(2.0 * marker.eta);
        const double radial_axis = marker.r + dt * marker.w;
        const double tangential_axis = dt * angular_momentum / marker.r;
        expected_r[i] = std::hypot(radial_axis, tangential_axis);
        expected_w[i] = (
            marker.w * radial_axis
            + angular_momentum / marker.r * tangential_axis) / expected_r[i];
    }
    venus_gpu::WpicState state(markers);
    venus_gpu::VlasovPoissonModule vp(mesh);
    venus_gpu::VlasovPoissonOptions options;
    options.gravity = 0.0;
    vp.advance_kdk(state, 0.0, dt, options);
    const std::vector<venus_gpu::Marker> after = state.download();
    double maximum_radius_error = 0.0;
    double maximum_velocity_error = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        maximum_radius_error = std::max(
            maximum_radius_error, std::abs(after[i].r - expected_r[i]));
        maximum_velocity_error = std::max(
            maximum_velocity_error, std::abs(after[i].w - expected_w[i]));
        assert(after[i].mass == markers[i].mass);
        assert(after[i].eta == markers[i].eta);
    }
    assert(maximum_radius_error < 2.0e-14);
    assert(maximum_velocity_error < 2.0e-14);
    std::cout << "ballistic_max_r_error=" << maximum_radius_error
              << " ballistic_max_w_error=" << maximum_velocity_error << '\n';
}

void test_origin_carrier_partition() {
    const venus_gpu::RadialMesh mesh = make_mesh(8, 0.5, 2.5);
    venus_gpu::Marker marker;
    marker.mass = 2.75;
    marker.r = 0.49;
    marker.w = 0.1;
    marker.eta = 0.02;
    marker.id = 1;
    venus_gpu::WpicState state(std::vector<venus_gpu::Marker>(1, marker));
    venus_gpu::VlasovPoissonModule vp(mesh);
    venus_gpu::VlasovPoissonOptions options;
    options.inner_boundary = venus_gpu::InnerBoundary::origin_parity;
    options.carry_active_mass_inside_inner_face = true;
    const venus_gpu::DepositionDiagnostics diagnostics =
        vp.deposit_and_solve(state, 0.0, options);
    assert(std::abs(
               diagnostics.resolved_active_mass
               + diagnostics.hidden_inner_mass - marker.mass)
           < 2.0e-14);
    assert(std::abs(
               diagnostics.deposited_mesh_mass
               - diagnostics.resolved_active_mass)
           < 2.0e-14);
    std::cout << "origin_resolved=" << diagnostics.resolved_active_mass
              << " hidden=" << diagnostics.hidden_inner_mass << '\n';
}

void test_exact_radial_origin_kdk() {
    const auto mesh=make_mesh(16,1,4);
    venus_gpu::VlasovPoissonOptions options;
    options.gravity=0;
    options.inner_boundary=venus_gpu::InnerBoundary::origin_parity;
    options.carry_active_mass_inside_inner_face=true;
    struct Case {double r,w,dt,expected_r,expected_w;};
    for (const Case& c : {Case{1,-1,1,0,1},Case{0,1,.4,.4,1},
                          Case{0,-1,.4,.4,1},Case{0,0,.4,0,0},
                          Case{1,-2,1,1,2},Case{0,1,5,3,-1}}) {
        venus_gpu::Marker marker;
        marker.mass=.25;marker.r=c.r;marker.w=c.w;marker.eta=0;marker.id=7;
        venus_gpu::WpicState state({marker});
        venus_gpu::VlasovPoissonModule vp(mesh);
        vp.advance_kdk(state,0,c.dt,options);
        auto after=state.download()[0];
        assert(std::abs(after.r-c.expected_r)<3e-14);
        assert(std::abs(after.w-c.expected_w)<3e-14);
        assert(after.mass==marker.mass && after.eta==0 && after.id==7);
        const auto diagnostics=vp.deposit_and_solve(state,c.dt,options);
        assert(std::abs(diagnostics.resolved_active_mass+diagnostics.hidden_inner_mass-.25)<3e-14);
        if (c.r>0) {
            after.w=-after.w;
            venus_gpu::WpicState reverse({after});
            vp.advance_kdk(reverse,0,c.dt,options);
            const auto back=reverse.download()[0];
            assert(std::abs(back.r-c.r)<3e-14 && std::abs(back.w+c.w)<3e-14);
        }
    }
    for (int invalid=0;invalid<3;++invalid) {
        venus_gpu::Marker marker;
        marker.mass=.25;marker.r=invalid==2?1:0;
        marker.w=invalid==2?-2:1;marker.eta=invalid==0?.1:0;marker.id=7;
        options.central_mass=invalid==0?0:.2;
        venus_gpu::WpicState state({marker});
        venus_gpu::VlasovPoissonModule vp(mesh);
        bool rejected=false;
        try {vp.advance_kdk(state,0,1,options);}
        catch(const std::runtime_error&){rejected=true;}
        assert(rejected);
    }
    std::cout << "GPU radial origin: exact crossing, endpoint, mass partition and reversal passed.\n";
}

void test_exact_physical_wall_events() {
    const auto mesh = make_mesh(16, 1.0, 4.0);
    // The third trajectory passes through the inner sphere and returns to
    // r>1 during one free drift. Checking the endpoint alone misses its bounce.
    struct Case { double r, w, L, dt; };
    const Case cases[] = {{1.1, -1.0, 0.5, 0.2}, {3.9, 0.8, 0.5, 0.3},
                          {1.1, -2.0, 0.2, 1.2}, {1.01, -0.4, 0.0, 0.1}};
    for (double alpha : {0.0, 0.5, 2.0 / 3.0}) {
        for (const Case& c : cases) {
            venus_gpu::VlasovPoissonOptions options;
            options.gravity = 0.0;
            options.inner_boundary = venus_gpu::InnerBoundary::reflecting_wall;
            options.outer_boundary = venus_gpu::OuterBoundary::reflecting_wall;
            options.expansion.enabled = alpha > 0.0;
            options.expansion.alpha = alpha;
            options.expansion.time_offset = 12.0;
            venus_gpu::Marker marker;
            marker.mass = 0.25; marker.r = c.r; marker.w = c.w;
            marker.eta = 0.5 * c.L * c.L; marker.id = 1;
            double expected_r = c.r, expected_w = c.w;
            const double tau = alpha > 0.0 ? 12.0 : 1.0 + c.dt;
            int events = 0;
            assert(ev::advance_central_drift_with_walls(expected_r, expected_w,
                c.L, c.dt, tau, alpha, 1.0, true, 4.0, true, &events));
            assert(events >= 1);
            venus_gpu::WpicState state({marker});
            venus_gpu::VlasovPoissonModule vp(mesh);
            vp.advance_kdk(state, 0.0, c.dt, options);
            const auto after = state.download().front();
            assert(std::abs(after.r - expected_r) < 3e-12);
            assert(std::abs(after.w - expected_w) < 3e-12);
            assert(after.mass == marker.mass && after.eta == marker.eta);
            assert(after.r >= 1.0 && after.r <= 4.0);
            // The same impact history integrated backward must recover the
            // initial state; expansion uses the endpoint cosmic time.
            double back_r = after.r, back_w = after.w;
            assert(ev::advance_central_drift_with_walls(back_r, back_w, c.L,
                -c.dt, tau + c.dt, alpha, 1.0, true, 4.0, true));
            assert(std::abs(back_r - c.r) < 5e-12);
            assert(std::abs(back_w - c.w) < 5e-12);
            if (alpha == 0.0) {
                const double initial_speed2 = c.w*c.w + c.L*c.L/(c.r*c.r);
                const double final_speed2 = after.w*after.w
                    + c.L*c.L/(after.r*after.r);
                assert(std::abs(final_speed2/initial_speed2 - 1.0) < 3e-13);
                // Independent Cartesian ray/sphere intersection and reflection.
                const double wall = c.w < 0.0 ? 1.0 : 4.0;
                const double discriminant = c.r*c.r*c.w*c.w
                    - initial_speed2*(c.r*c.r-wall*wall);
                const double impact_time = (-c.r*c.w
                    + (c.w < 0.0 ? -1.0 : 1.0)*std::sqrt(discriminant))
                    / initial_speed2;
                const double x = c.r + impact_time*c.w;
                const double y = impact_time*c.L/c.r;
                const double normal_speed = (x*c.w+y*c.L/c.r)/wall;
                const double vx = c.w-2.0*normal_speed*x/wall;
                const double vy = c.L/c.r-2.0*normal_speed*y/wall;
                const double final_x = x+(c.dt-impact_time)*vx;
                const double final_y = y+(c.dt-impact_time)*vy;
                const double exact_r = std::hypot(final_x,final_y);
                const double exact_w = (final_x*vx+final_y*vy)/exact_r;
                assert(std::abs(after.r-exact_r) < 3e-12);
                assert(std::abs(after.w-exact_w) < 3e-12);
            }
        }
    }
    for (double alpha : {0.0, 1.0}) {
        for (double wall : {1.0, 4.0}) {
            // Both backgrounds admit a particle at rest in physical radius.
            // Its peculiar velocity cancels H*r when alpha=1.
            double r = wall, w = -alpha * wall / 2.0;
            int events = -1;
            assert(ev::advance_central_drift_with_walls(r, w, 0.0, 1.0,
                2.0, alpha, 1.0, true, 4.0, true, &events));
            assert(r == wall && w == -alpha * wall / 3.0 && events == 0);
            venus_gpu::Marker marker;
            marker.mass = 1.0; marker.r = wall; marker.w = -alpha * wall / 2.0;
            marker.id = 1;
            venus_gpu::WpicState stationary({marker});
            venus_gpu::VlasovPoissonModule module(mesh);
            venus_gpu::VlasovPoissonOptions zero;
            zero.gravity = 0.0;
            zero.expansion.enabled = alpha != 0.0;
            zero.expansion.alpha = alpha;
            zero.expansion.time_offset = 2.0;
            module.advance_kdk(stationary, 0.0, 1.0, zero);
            const auto result = stationary.download().front();
            assert(result.r == wall && result.w == -alpha * wall / 3.0);
        }
    }
    // Eight complete static-wall impacts return this radial ray to its start.
    venus_gpu::Marker marker;
    marker.mass=1.0; marker.r=2.0; marker.w=240.0; marker.id=2;
    venus_gpu::VlasovPoissonOptions options; options.gravity=0.0;
    venus_gpu::WpicState state({marker}); venus_gpu::VlasovPoissonModule vp(mesh);
    vp.advance_kdk(state,0.0,0.1,options);
    const auto after=state.download().front();
    assert(std::abs(after.r-2.0)<5e-12 && std::abs(after.w-240.0)<5e-12);
    std::cout << "Exact physical wall events: angular energy, hidden inner "
              << "crossing, expansion, and signed host reversal passed.\n";
}

void test_inner_face_force_continuity() {
    const auto mesh = make_mesh(8, 1.0, 5.0);
    venus_gpu::VlasovPoissonOptions options;
    options.inner_boundary=venus_gpu::InnerBoundary::origin_parity;
    options.carry_active_mass_inside_inner_face=true;
    for (double core_mass : {0.0, 2.0}) {
        for (double central_mass : {0.0, 3.0}) {
            options.central_mass=central_mass;
            venus_gpu::FieldHost moments;
            const auto zeros=std::vector<double>(mesh.cells(),0.0);
            moments.shell_mass=zeros;moments.shell_mass[0]=1.0;
            moments.shell_radial_momentum=zeros;
            moments.shell_radial_second_moment=zeros;
            moments.shell_tangential_second_moment=zeros;
            moments.shell_radial_energy_flux=zeros;
            moments.shell_mass_square=zeros;moments.shell_mass_square[0]=1.0;
            moments.diagnostics.resolved_active_mass=1.0;
            moments.diagnostics.deposited_mesh_mass=1.0;
            moments.diagnostics.hidden_inner_mass=core_mass;
            venus_gpu::VlasovPoissonModule vp(mesh);
            vp.solve_from_global_moments(moments,0.0,options);
            for (double radius : {0.25,0.75,1.0-1e-8,1.0,1.0+1e-8}) {
                venus_gpu::Marker probe;probe.mass=1.0;probe.r=radius;probe.id=1;
                venus_gpu::WpicState state({probe});
                // The first half kick directly measures the initial force.
                // Zero angular momentum leaves w unchanged during this drift.
                constexpr double dt=1e-6;
                vp.begin_kdk_with_current_field(state,0.0,dt,options);
                const auto after=state.download().front();
                const double measured=-2.0*after.w/dt;
                const double expected=radius<1.0
                    ? central_mass/(radius*radius)+core_mass*radius
                    : (central_mass+core_mass+(radius-1.0)/(mesh.edge[1]-1.0))
                        /(radius*radius);
                assert(std::abs(measured-expected)<2e-12*std::max(1.0,std::abs(expected)));
            }
        }
    }
    std::cout<<"Inner-face gravity: empty cavity, uniform core, point mass, "
             <<"and face continuity passed.\n";
}

void test_homogeneous_background_source() {
    venus_gpu::VlasovPoissonOptions options;
    options.gravity=1.0;options.inner_boundary=venus_gpu::InnerBoundary::origin_parity;
    options.expansion.enabled=true;options.expansion.alpha=2.0/3.0;
    options.expansion.time_offset=12.0;options.subtract_homogeneous_background=true;
    options.carry_active_mass_inside_inner_face=true;
    const double pi=std::acos(-1.0),H=options.expansion.alpha/12.0;
    const double rho=3.0*H*H/(8.0*pi);
    const double core_background=4.0*pi*rho/3.0;
    auto moments_for=[&](const venus_gpu::RadialMesh& mesh,bool exact_volume) {
        venus_gpu::FieldHost m;const auto zeros=std::vector<double>(mesh.cells(),0.0);
        m.shell_mass=zeros;m.shell_radial_momentum=zeros;
        m.shell_radial_second_moment=zeros;m.shell_tangential_second_moment=zeros;
        m.shell_radial_energy_flux=zeros;m.shell_mass_square=zeros;
        for(std::size_t i=0;i<mesh.cells();++i) {
            m.shell_mass[i]=rho*(exact_volume?mesh.shell_volume[i]:
                4.0*pi*mesh.radius_center[i]*mesh.radius_center[i]
                *(mesh.edge[i+1]-mesh.edge[i]));
            m.shell_mass_square[i]=m.shell_mass[i]*m.shell_mass[i];
            m.diagnostics.resolved_active_mass+=m.shell_mass[i];
        }
        m.diagnostics.deposited_mesh_mass=m.diagnostics.resolved_active_mass;
        m.diagnostics.hidden_inner_mass=core_background;return m;
    };
    const auto mesh=make_mesh(8,1.0,5.0);
    venus_gpu::VlasovPoissonModule vp(mesh);
    auto moments=moments_for(mesh,false);
    vp.solve_from_global_moments(moments,0.0,options);
    auto field=vp.download_field();
    for(double source:field.source_mass_edge)assert(std::abs(source)<3e-16);
    for(double force:field.phi_r)assert(std::abs(force)<3e-16);
    assert(field.diagnostics.hidden_inner_mass==core_background);
    // An empty physical central sphere has negative density contrast. The
    // physical mass ledger remains zero while its source is -M_background.
    moments.diagnostics.hidden_inner_mass=0.0;
    vp.solve_from_global_moments(moments,0.0,options);field=vp.download_field();
    for(double source:field.source_mass_edge)assert(std::abs(source+core_background)<3e-16);
    assert(field.diagnostics.hidden_inner_mass==0.0);
    venus_gpu::Marker marker;marker.mass=1.0;marker.r=0.5;marker.id=1;
    venus_gpu::WpicState state({marker});constexpr double dt=1e-5;
    vp.begin_kdk_with_current_field(state,0.0,dt,options);
    // For L=0 the expanding free drift rescales the half-kicked w by a0/a1.
    const double a1=std::pow((12.0+dt)/12.0,2.0/3.0);
    const double measured_force=-2.0*state.download().front().w*a1/dt;
    assert(std::abs(measured_force+core_background*marker.r)<3e-16);
    // The public shell density uses exact volume, while source reconstruction
    // uses midpoint line-density quadrature. Its uniform-background residual
    // converges as Delta r^2 at fixed domain boundaries.
    double previous=0.0;
    for(int n:{8,16,32}) {
        const auto refined=make_mesh(n,1.0,5.0);
        venus_gpu::VlasovPoissonModule module(refined);
        module.solve_from_global_moments(moments_for(refined,true),0.0,options);
        const double residual=module.download_field().source_mass_edge.back();
        assert(residual>0.0);
        if(previous>0.0)assert(std::abs(residual/previous-0.25)<2e-11);
        std::cout << "Uniform exact-volume source: cells=" << n
                  << " residual_mass=" << residual;
        if (previous > 0.0) std::cout << " refinement_ratio=" << residual/previous;
        std::cout << '\n';
        previous=residual;
    }
    // A density-contrast input is already background-subtracted.
    options.subtract_homogeneous_background=false;
    auto empty=moments_for(mesh,false);
    std::fill(empty.shell_mass.begin(),empty.shell_mass.end(),0.0);
    std::fill(empty.shell_mass_square.begin(),empty.shell_mass_square.end(),0.0);
    empty.diagnostics={};vp.solve_from_global_moments(empty,0.0,options);
    field=vp.download_field();for(double source:field.source_mass_edge)assert(source==0.0);
    std::cout<<"Homogeneous total-density source and empty-cavity contrast passed; "
             <<"exact-volume residual decreases by four under doubled resolution.\n";
}

void test_option_validation() {
    const venus_gpu::RadialMesh mesh = make_mesh(8, 0.5, 2.5);
    venus_gpu::WpicState empty;
    venus_gpu::VlasovPoissonModule vp(mesh);
    auto rejected = [&](const venus_gpu::VlasovPoissonOptions& options) {
        try {
            (void)vp.deposit_and_solve(empty, 0.0, options);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };

    venus_gpu::VlasovPoissonOptions options;
    options.particle_shape = static_cast<venus_gpu::ParticleShape>(99);
    assert(rejected(options));
    options = {};
    options.inner_boundary = static_cast<venus_gpu::InnerBoundary>(99);
    assert(rejected(options));
    options = {};
    options.outer_boundary = static_cast<venus_gpu::OuterBoundary>(99);
    assert(rejected(options));
    options = {};
    options.expansion.enabled = true;
    options.expansion.alpha = std::numeric_limits<double>::quiet_NaN();
    assert(rejected(options));
    options.expansion.alpha = -0.5;
    assert(rejected(options));
    options = {};
    options.gravity = 0.0;
    options.expansion.enabled = true;
    options.expansion.alpha = 2.0 / 3.0;
    options.expansion.time_offset = 12.0;
    options.subtract_homogeneous_background = true;
    assert(rejected(options));
    // Zero gravity remains a valid ballistic expanding calculation when the
    // input masses already represent density contrast.
    options.subtract_homogeneous_background = false;
    (void)vp.deposit_and_solve(empty, 0.0, options);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc == 2 ? argv[1] : "";
    if (argc > 2
        || (!mode.empty() && mode != "--option-validation-only")) {
        std::cerr << "usage: " << argv[0]
                  << " [--option-validation-only]\n";
        return 2;
    }
    venus_gpu::require_cuda_device();
    if (mode == "--option-validation-only") {
        test_option_validation();
        std::cout << "WPIC Vlasov-Poisson option validation passed.\n";
        return 0;
    }
    test_marker_upload_validation();
    test_mass_conservative_cic_and_poisson();
    test_ballistic_exact_central_drift();
    test_origin_carrier_partition();
    test_exact_radial_origin_kdk();
    test_exact_physical_wall_events();
    test_option_validation();
    test_inner_face_force_continuity();
    test_homogeneous_background_source();
    std::cout << "WPIC Vlasov-Poisson GPU tests passed.\n";
    return 0;
}
