// Construct a compact, finite-grid distribution for collision examples.
// A reference spherical potential selects bound orbital energies. The production
// solver reconstructs the self-consistent field from the saved distribution.
// This small input exercises numerical operators; it is a convergence starting
// point for learning the interface rather than a relaxed astrophysical halo.
#include "ev/solver.hpp"
#include <iostream>

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

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: generate_collision_example PATH\n"; return 2; }
    ev::Params p;
    p.physics_model="boltzmann_poisson";
    p.vp_universe="static";
    p.vp_interpolation="conservative_sldg";
    p.vp_timestep_mode="frank_dynamical";
    p.kinetic_grid="cell_average_eta";
    p.kinetic_eta_power=6;
    p.Nr=3; p.Nw=5; p.NL=2;
    p.r_min=0.5; p.r_max=2; p.w_max=2; p.L_max=2;
    p.w_stretch=0; p.w_center=0;
    p.inner_boundary="reflecting";
    p.outer_boundary="outflow_vacuum";
    p.momentum_boundary="outflow_vacuum";
    p.vp_sldg_subcells=1;
    const auto geometry=ev::make_cell_average_eta_geometry(ev::make_positive_radial_cell_axis(p),p);
    const auto field=represented_test_field(geometry,p);
    auto f=represented_energy_equilibrium(geometry,p,field,1.2,0.6,1.8);
    for(auto& value:f) value *= 1.0e-4;
    const std::filesystem::path path=argv[1];
    std::filesystem::create_directories(path.parent_path());
    ev::write_restart_state(path,path.parent_path(),p,0.0,0,f);
    std::cout << "Saved compact collision example to " << path << '\n';
}
