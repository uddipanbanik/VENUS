#include "venus_gpu/wpic.hpp"
#include "venus_gpu/restart.hpp"

#include "ev/weighted_vlasov_pic.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <cstdio>
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

double relative_error(double first, double second) {
    return std::abs(first - second)
        / std::max({1.0, std::abs(first), std::abs(second)});
}

void check_close(
    const std::vector<double>& gpu,
    const std::vector<double>& cpu,
    double tolerance,
    const char* label) {
    assert(gpu.size() == cpu.size());
    double maximum = 0.0;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        maximum = std::max(maximum, relative_error(gpu[i], cpu[i]));
    }
    std::cerr << label << "_max_relative_error=" << maximum << '\n';
    assert(maximum < tolerance);
}

ev::Params current_expanding_parameters() {
    ev::Params p;
    p.physics_model = "vlasov_poisson";
    p.vp_universe = "expanding";
    p.vp_alpha = 2.0 / 3.0;
    p.vp_time_offset = 12.0;
    p.vp_gravity = 1.0;
    p.vp_poisson_source = "density_contrast";
    p.vp_poisson_reconstruction = "constant_line_mass";
    p.vp_interpolation = "hamiltonian_sldg";
    p.vp_orbit_regularization = "off";
    p.kinetic_grid = "cell_average_eta";
    p.kinetic_eta_grid = "uniform";
    p.Nr = 96;
    p.Nw = 4;
    p.NL = 2;
    p.r_min = 0.1;
    p.r_max = 1000.0;
    p.radial_grid = "log";
    p.w_max = 3.0;
    p.L_max = 8.0;
    p.inner_boundary = "origin_parity";
    p.outer_boundary = "reflecting";
    p.integrator = "semi_lagrangian_kick_drift_kick";
    return p;
}

venus_gpu::VlasovPoissonOptions gpu_options() {
    venus_gpu::VlasovPoissonOptions options;
    options.gravity = 1.0;
    options.particle_shape = venus_gpu::ParticleShape::tsc;
    options.inner_boundary = venus_gpu::InnerBoundary::origin_parity;
    options.outer_boundary = venus_gpu::OuterBoundary::reflecting_wall;
    options.expansion.enabled = true;
    options.expansion.alpha = 2.0 / 3.0;
    options.expansion.time_offset = 12.0;
    // The current CPU runner selects vp_poisson_source=density_contrast: its
    // marker masses are already the source contrast.  Consequently neither
    // implementation subtracts rho_bar again inside the Poisson solve.
    options.subtract_homogeneous_background = false;
    options.carry_active_mass_inside_inner_face = true;
    return options;
}

void run_current_cpu_gpu_parity() {
    const ev::Params p = current_expanding_parameters();
    const ev::WeightedPICFieldGrid cpu_grid =
        ev::make_weighted_pic_field_grid(p, p.Nr);
    const venus_gpu::RadialMesh gpu_mesh =
        venus_gpu::RadialMesh::from_edges(cpu_grid.mesh.edge);

    constexpr std::size_t count = 240000;
    std::vector<ev::WeightedPICMarker> cpu_markers(count);
    std::vector<venus_gpu::Marker> gpu_markers(count);
    const double q0 = std::pow(p.r_min, 3) / 3.0;
    const double q1 = std::pow(p.r_max, 3) / 3.0;
    for (std::size_t i = 0; i < count; ++i) {
        // The nonlinear q mapping fills all TSC stencils, while a small
        // deterministic inner population exercises the hidden carrier.
        double radius = std::cbrt(
            3.0 * (q0 + (q1 - q0) * uniform(i + 11)));
        if (i < 1024) {
            radius = p.r_min * (0.15 + 0.8 * uniform(i + 29));
        }
        ev::WeightedPICMarker cpu;
        cpu.mass = (0.5 + uniform(i + 101)) / count;
        cpu.r = radius;
        cpu.w = 0.4 * (2.0 * uniform(i + 211) - 1.0);
        cpu.eta = 0.02 + 0.2 * uniform(i + 307);
        cpu.id = i;
        cpu.source_cell = i % static_cast<std::size_t>(p.Nr);
        cpu.internal_energy = 0.01 * cpu.mass;
        cpu.internal_velocity_dispersion = std::sqrt(0.02 / 3.0);
        cpu_markers[i] = cpu;

        venus_gpu::Marker gpu;
        gpu.mass = cpu.mass;
        gpu.r = cpu.r;
        gpu.w = cpu.w;
        gpu.eta = cpu.eta;
        gpu.id = cpu.id;
        gpu.source_cell = cpu.source_cell;
        gpu.internal_energy = cpu.internal_energy;
        gpu.internal_velocity_dispersion =
            cpu.internal_velocity_dispersion;
        gpu_markers[i] = gpu;
    }

    constexpr double time = 37.25;
    const ev::WeightedPICDepositedMoments cpu_deposit =
        ev::deposit_weighted_vlasov_pic_parallel(
            cpu_markers,
            cpu_grid.mesh,
            ev::WeightedPICParticleShape::tsc,
            4,
            true);
    const ev::VlasovPoissonField cpu_field =
        ev::solve_weighted_pic_poisson_field(
            cpu_deposit, cpu_grid.mesh, time, p);

    venus_gpu::WpicState gpu_state(gpu_markers);
    venus_gpu::VlasovPoissonModule gpu_vp(gpu_mesh);
    const venus_gpu::VlasovPoissonOptions options = gpu_options();
    const venus_gpu::DepositionDiagnostics gpu_diagnostics =
        gpu_vp.deposit_and_solve(gpu_state, time, options);
    const venus_gpu::FieldHost gpu_field = gpu_vp.download_field();

    check_close(gpu_field.shell_mass, cpu_deposit.shell_mass, 2.0e-12,
        "shell_mass");
    check_close(gpu_field.shell_radial_momentum,
        cpu_deposit.shell_radial_momentum, 2.0e-12, "radial_momentum");
    check_close(gpu_field.source_mass_edge, cpu_field.source_mass_edge,
        2.0e-12, "source_mass_edge");
    check_close(gpu_field.source_line_density,
        cpu_field.source_line_density, 2.0e-12, "source_line_density");
    check_close(gpu_field.phi, cpu_field.phi, 2.0e-12, "phi");
    check_close(gpu_field.phi_r, cpu_field.phi_r, 2.0e-12, "phi_r");
    assert(relative_error(
               gpu_diagnostics.hidden_inner_mass,
               cpu_deposit.hidden_inner_mass)
           < 2.0e-12);

    constexpr double dt = 0.015;
    ev::begin_weighted_pic_self_consistent_kdk_parallel(
        cpu_markers,
        time,
        dt,
        cpu_grid.radial_coordinate,
        cpu_grid.radius,
        cpu_field,
        p,
        4);
    const ev::WeightedPICDepositedMoments endpoint_deposit =
        ev::deposit_weighted_vlasov_pic_parallel(
            cpu_markers,
            cpu_grid.mesh,
            ev::WeightedPICParticleShape::tsc,
            4,
            true);
    const ev::VlasovPoissonField endpoint_field =
        ev::solve_weighted_pic_poisson_field(
            endpoint_deposit, cpu_grid.mesh, time + dt, p);
    ev::finish_weighted_pic_self_consistent_kdk_parallel(
        cpu_markers,
        time + dt,
        dt,
        cpu_grid.radial_coordinate,
        cpu_grid.radius,
        endpoint_field,
        p,
        4);

    gpu_vp.advance_kdk(gpu_state, time, dt, options);
    const std::vector<venus_gpu::Marker> after = gpu_state.download();
    double maximum_radius_error = 0.0;
    double maximum_velocity_error = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        maximum_radius_error = std::max(
            maximum_radius_error,
            relative_error(after[i].r, cpu_markers[i].r));
        maximum_velocity_error = std::max(
            maximum_velocity_error,
            relative_error(after[i].w, cpu_markers[i].w));
        assert(after[i].mass == cpu_markers[i].mass);
        assert(after[i].eta == cpu_markers[i].eta);
    }
    std::cout << "kdk_radius_max_relative_error=" << maximum_radius_error
              << " kdk_velocity_max_relative_error="
              << maximum_velocity_error << '\n';
    assert(maximum_radius_error < 3.0e-12);
    assert(maximum_velocity_error < 3.0e-12);
}

void run_wall_cpu_gpu_parity() {
    struct Case { double r, w, L, dt; };
    const Case cases[]={{1.1,-1.0,0.5,0.2},{3.9,0.8,0.5,0.3},
                        {1.1,-2.0,0.2,1.2},{1.01,-0.4,0.0,0.1}};
    for(double alpha : {0.0,0.5,2.0/3.0}) {
        ev::Params p=current_expanding_parameters();
        p.vp_universe=alpha>0.0?"expanding":"static";p.vp_alpha=alpha;
        p.r_min=1.0;p.r_max=4.0;p.Nr=16;p.inner_boundary="reflecting";
        const auto grid=ev::make_weighted_pic_field_grid(p,p.Nr);
        const auto mesh=venus_gpu::RadialMesh::from_edges(grid.mesh.edge);
        // A zero-mass field isolates the drift and boundary event from kicks.
        const auto empty=ev::deposit_weighted_vlasov_pic_parallel(
            {},grid.mesh,ev::WeightedPICParticleShape::tsc,1,false);
        const auto zero=ev::solve_weighted_pic_poisson_field(empty,grid.mesh,0.0,p);
        for(const Case& c:cases) {
            ev::WeightedPICMarker m;m.mass=1.0;m.r=c.r;m.w=c.w;
            m.eta=0.5*c.L*c.L;m.id=1;
            std::vector<ev::WeightedPICMarker> cpu{m};
            ev::begin_weighted_pic_self_consistent_kdk_parallel(cpu,0.0,c.dt,
                grid.radial_coordinate,grid.radius,zero,p,1);
            ev::finish_weighted_pic_self_consistent_kdk_parallel(cpu,c.dt,c.dt,
                grid.radial_coordinate,grid.radius,zero,p,1);
            venus_gpu::Marker gm;gm.mass=m.mass;gm.r=m.r;gm.w=m.w;
            gm.eta=m.eta;gm.id=m.id;
            auto o=gpu_options();o.gravity=0.0;o.inner_boundary=venus_gpu::InnerBoundary::reflecting_wall;
            o.expansion.enabled=alpha>0.0;o.expansion.alpha=alpha;
            venus_gpu::WpicState gpu({gm});venus_gpu::VlasovPoissonModule vp(mesh);
            vp.advance_kdk(gpu,0.0,c.dt,o);const auto result=gpu.download().front();
            assert(relative_error(result.r,cpu[0].r)<3e-12);
            assert(relative_error(result.w,cpu[0].w)<3e-12);
            assert(result.eta==cpu[0].eta && result.mass==cpu[0].mass);
        }
    }
    std::cout<<"CPU/GPU exact inner/outer wall event parity passed.\n";
}

void run_restart_interoperability() {
    const std::string cpu_path = "cpu_written_gpu_restart_test.bin";
    const std::string gpu_path = "gpu_written_cpu_restart_test.bin";
    ev::WeightedPICRestartState cpu;
    cpu.time = 50.125;
    cpu.step = 4123;
    cpu.sampler = {UINT64_C(0x1234), 77};
    cpu.collision_sampler = {UINT64_C(0x5678), 91};
    cpu.potential_event_age = 0.25;
    cpu.collision_event_age = 0.375;
    cpu.collision_timescale_estimate = 2.5;
    cpu.potential_timescale_estimate = 3.75;
    for (std::uint64_t i = 0; i < 17; ++i) {
        ev::WeightedPICMarker marker;
        marker.mass = 0.01 * (i + 1);
        marker.r = 0.2 + i;
        marker.w = 0.03 * static_cast<double>(i) - 0.2;
        marker.eta = 0.001 * (i + 1);
        marker.internal_energy = 0.002 * (i + 1);
        marker.internal_velocity_dispersion = std::sqrt(
            2.0 * marker.internal_energy / (3.0 * marker.mass));
        marker.id = 1000 + i;
        marker.source_cell = i % 4;
        cpu.markers.push_back(marker);
    }
    ev::write_weighted_pic_restart(cpu_path, cpu);
    venus_gpu::RestartState gpu = venus_gpu::read_restart(cpu_path);
    assert(gpu.time == cpu.time);
    assert(gpu.step == cpu.step);
    assert(gpu.initialization_seed == cpu.sampler.seed);
    assert(gpu.initialization_counter == cpu.sampler.next_counter);
    assert(gpu.collision_sampler.seed == cpu.collision_sampler.seed);
    assert(gpu.collision_sampler.next_counter
        == cpu.collision_sampler.next_counter);
    assert(gpu.markers.size() == cpu.markers.size());
    venus_gpu::write_restart(gpu_path, gpu);
    const ev::WeightedPICRestartState roundtrip =
        ev::read_weighted_pic_restart(gpu_path);
    assert(roundtrip.time == cpu.time);
    assert(roundtrip.step == cpu.step);
    assert(roundtrip.potential_event_age == cpu.potential_event_age);
    assert(roundtrip.collision_event_age == cpu.collision_event_age);
    assert(roundtrip.collision_timescale_estimate
        == cpu.collision_timescale_estimate);
    assert(roundtrip.markers.size() == cpu.markers.size());
    for (std::size_t i = 0; i < cpu.markers.size(); ++i) {
        assert(roundtrip.markers[i].mass == cpu.markers[i].mass);
        assert(roundtrip.markers[i].r == cpu.markers[i].r);
        assert(roundtrip.markers[i].w == cpu.markers[i].w);
        assert(roundtrip.markers[i].eta == cpu.markers[i].eta);
        assert(roundtrip.markers[i].id == cpu.markers[i].id);
        assert(roundtrip.markers[i].internal_energy
            == cpu.markers[i].internal_energy);
    }
    std::remove(cpu_path.c_str());
    std::remove(gpu_path.c_str());
    std::cout << "CPU_GPU_restart_v6_interoperability=passed\n";
}

}  // namespace

int main() {
    venus_gpu::require_cuda_device();
    run_restart_interoperability();
    run_current_cpu_gpu_parity();
    run_wall_cpu_gpu_parity();
    std::cout << "VENUS CPU/GPU Vlasov-Poisson parity passed.\n";
    return 0;
}
