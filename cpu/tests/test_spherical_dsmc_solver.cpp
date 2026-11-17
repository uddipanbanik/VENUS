// Integration tests for persistent SphDSMC gravity, timestep/shell locality,
// restart determinism, stiff-orbit retry, and global conservation ledgers.

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "ev/config.hpp"
#include "ev/spherical_dsmc_solver.hpp"

namespace {

class EnvironmentRestorer {
public:
    explicit EnvironmentRestorer(const char* name)
        : name_(name) {
        const char* value = std::getenv(name);
        if (value != nullptr) {
            had_value_ = true;
            value_ = value;
        }
    }

    ~EnvironmentRestorer() {
        if (had_value_) {
            ::setenv(name_.c_str(), value_.c_str(), 1);
        } else {
            ::unsetenv(name_.c_str());
        }
    }

private:
    std::string name_;
    std::string value_;
    bool had_value_ = false;
};

bool nearly_equal(double a, double b, double tolerance = 1.0e-12) {
    return std::abs(a - b)
        <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}

std::vector<ev::SphericalDSMCParticle> circular_markers(
    std::size_t count,
    double particle_mass,
    const ev::Params& p) {
    std::vector<ev::SphericalDSMCParticle> particles(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double radius = 0.7 + 0.02 * static_cast<double>(i);
        const double enclosed =
            p.central_mass + (static_cast<double>(i) + 0.5) * particle_mass;
        particles[i] = {
            radius,
            0.0,
            std::sqrt(p.vp_gravity * enclosed * radius),
            static_cast<std::uint64_t>(i),
        };
    }
    return particles;
}

void test_circular_orbits_are_fixed_points() {
    ev::Params p;
    p.central_mass = 1.0;
    p.vp_gravity = 1.0;
    p.threads = 4;
    p.vp_orbit_regularization = "off";
    const double particle_mass = 1.0e-6;
    std::vector<ev::SphericalDSMCParticle> particles =
        circular_markers(128, particle_mass, p);
    const std::vector<ev::SphericalDSMCParticle> before = particles;
    const ev::SphericalDSMCEnergy energy_before =
        ev::spherical_dsmc_total_energy(particles, particle_mass, p);

    const ev::SphericalDSMCDiagnostics diagnostics =
        ev::spherical_dsmc_gravity_step(particles, particle_mass, 0.4, p);
    const ev::SphericalDSMCEnergy energy_after =
        ev::spherical_dsmc_total_energy(particles, particle_mass, p);

    assert(diagnostics.lc_retries == 0);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        assert(nearly_equal(particles[i].r, before[i].r, 2.0e-14));
        assert(nearly_equal(particles[i].vr, before[i].vr, 2.0e-14));
        assert(nearly_equal(particles[i].L, before[i].L, 2.0e-14));
        assert(particles[i].id == before[i].id);
    }
    assert(nearly_equal(energy_after.total, energy_before.total, 2.0e-14));
}

void test_origin_endpoint_failure_restores_particles() {
    EnvironmentRestorer restore_paper("EV_SPHERICAL_DSMC_PAPER_EXACT");
    ::unsetenv("EV_SPHERICAL_DSMC_PAPER_EXACT");
    ev::Params p;
    p.vp_gravity = 1.0;
    p.central_mass = 0.0;
    p.bp_dsmc_gravity_eta = 0.01;
    p.dt_max = 0.2;
    p.threads = 1;
    // The actual acceleration-based selector gives dt=0.2. The half kick
    // changes vr to -5, so the drift lands exactly at the origin. The
    // spherical shell force is singular there; every supported orbit mode
    // must preserve the accepted input if it cannot resolve that passage.
    const std::vector<ev::SphericalDSMCParticle> initial = {
        {1.0, -4.95, 0.0, 0},
        {10.0, 0.0, std::sqrt(15.0), 1},
    };
    for (const std::string mode : {"off", "auto", "always"}) {
        p.vp_orbit_regularization = mode;
        auto particles = initial;
        const double dt = std::min(p.dt_max,
            ev::spherical_dsmc_gravity_timestep(particles, 1.0, p));
        assert(nearly_equal(dt, 0.2, 1e-15));
        bool rejected = false;
        try {
            ev::spherical_dsmc_gravity_step(particles, 1.0, dt, p);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("input state restored")
                != std::string::npos;
        }
        assert(rejected);
        assert(particles.size() == initial.size());
        for (std::size_t i = 0; i < initial.size(); ++i) {
            assert(particles[i].r == initial[i].r);
            assert(particles[i].vr == initial[i].vr);
            assert(particles[i].L == initial[i].L);
            assert(particles[i].mass == initial[i].mass);
            assert(particles[i].id == initial[i].id);
        }
    }

    // A finite step crossing the origin retains radial parity and leaves
    // angular momentum invariant. This separately checks the accepted branch.
    p.vp_orbit_regularization = "off";
    std::vector<ev::SphericalDSMCParticle> crossing = {{1.0, -5.95, 0.0, 0}};
    ev::spherical_dsmc_gravity_step(crossing, 1.0, 0.2, p);
    assert(nearly_equal(crossing[0].r, 0.2));
    assert(nearly_equal(crossing[0].vr, 4.75));
    assert(crossing[0].L == 0.0);
}

void test_frank_equation5_particle_potential() {
    ev::Params p;
    p.vp_gravity = 1.0;
    p.central_mass = 0.0;
    std::vector<ev::SphericalDSMCParticle> particles = {
        {1.0, 0.0, 0.0, 0},
        {2.0, 0.0, 0.0, 1},
        {4.0, 0.0, 0.0, 2},
    };
    const std::vector<double> potential =
        ev::spherical_dsmc_specific_potential(particles, 2.0, p);
    assert(potential.size() == 3);
    assert(nearly_equal(potential[0], -3.5));
    assert(nearly_equal(potential[1], -2.5));
    assert(nearly_equal(potential[2], -1.5));
}

void test_force_consistent_orbit_audit_removes_irreducible_half_self() {
    EnvironmentRestorer restore(
        "EV_SPHERICAL_DSMC_FORCE_CONSISTENT_ORBIT_AUDIT");
    ::setenv("EV_SPHERICAL_DSMC_FORCE_CONSISTENT_ORBIT_AUDIT", "1", 1);
    ev::Params p;
    p.vp_gravity = 1.0;
    p.central_mass = 0.0;
    constexpr double particle_mass = 1.1252580933331495e-6;
    constexpr double r0 = 8.7060951333325921e-6;
    constexpr double vr0 = -1.1283936866763409;
    constexpr double angular_momentum = 2.6789547127091404e-6;
    constexpr double rf = 2.0116092656067885e-6;
    constexpr double vrf = -0.15665364678865304;
    std::vector<ev::SphericalDSMCParticle> before = {
        {r0, vr0, angular_momentum, 0},
    };
    std::vector<ev::SphericalDSMCParticle> after = {
        {rf, vrf, angular_momentum, 0},
    };
    std::vector<double> outer_sum;
    std::vector<double> potential_before;
    std::vector<double> potential_after;
    ev::spherical_dsmc_orbit_audit_specific_potential(
        before, particle_mass, p, outer_sum, potential_before);
    ev::spherical_dsmc_orbit_audit_specific_potential(
        after, particle_mass, p, outer_sum, potential_after);
    const double energy_before =
        ev::spherical_dsmc_specific_kinetic_energy(before[0])
        + potential_before[0];
    const double energy_after =
        ev::spherical_dsmc_specific_kinetic_energy(after[0])
        + potential_after[0];

    // These are the actual initial/final states from the failed central LC
    // orbit.  They conserve the equation-(4) Kepler invariant to roundoff;
    // equation (5) alone would leave an O(0.2) half-self residual.
    assert(std::abs(energy_after - energy_before) < 5.0e-8);
    const std::vector<double> literal_before =
        ev::spherical_dsmc_specific_potential(before, particle_mass, p);
    const std::vector<double> literal_after =
        ev::spherical_dsmc_specific_potential(after, particle_mass, p);
    const double literal_change =
        ev::spherical_dsmc_specific_kinetic_energy(after[0])
        + literal_after[0]
        - ev::spherical_dsmc_specific_kinetic_energy(before[0])
        - literal_before[0];
    assert(nearly_equal(literal_change, -0.2150662445532475, 1.0e-12));
}

void test_paper_orbit_energy_defect_parabolic_guard() {
    const double literal = ev::spherical_dsmc_paper_orbit_energy_defect(
        -0.49, -0.50, 1.50, 0.01);
    assert(nearly_equal(literal, 0.02));

    // A collision may place a particle arbitrarily close to E=0 while K and
    // |Phi| remain order unity.  The literal |Delta E/E| is singular there;
    // the guarded defect remains a stringent absolute 1e-4 energy-scale test.
    const double parabolic = ev::spherical_dsmc_paper_orbit_energy_defect(
        1.0e-6, 1.0e-12, 2.0, 0.01);
    assert(parabolic < 1.0e-4);
}

void test_lc_retry_bookkeeping_is_retry_local() {
    ev::Params p;
    p.central_mass = 1.0;
    p.vp_gravity = 1.0;
    p.threads = 4;
    p.vp_orbit_regularization = "always";
    p.vp_orbit_energy_tolerance = 1.0e-10;
    p.vp_orbit_lc_relative_tolerance = 1.0e-11;
    p.vp_orbit_lc_absolute_tolerance = 1.0e-13;
    p.vp_orbit_lc_max_steps = 4096;
    const double particle_mass = 1.0e-7;
    std::vector<ev::SphericalDSMCParticle> particles =
        circular_markers(48, particle_mass, p);

    const ev::SphericalDSMCDiagnostics diagnostics =
        ev::spherical_dsmc_gravity_step(particles, particle_mass, 0.2, p);
    assert(diagnostics.lc_retries == 48);
    assert(diagnostics.lc_accepted == 48);
    assert(diagnostics.lc_unresolved == 0);
}

void test_paper_lc_leapfrog_converges_through_pericentre() {
    constexpr double angular_momentum_squared = 0.1;
    constexpr double eccentricity = 0.9;
    constexpr double semimajor_axis = 1.0 / (1.0 + eccentricity);
    const double half_period = ev::PI
        * semimajor_axis * std::sqrt(semimajor_axis);
    auto rhs = [](const ev::CharacteristicState& state) {
        const double radius = std::max(state.r, 1.0e-300);
        return ev::CharacteristicDerivative{
            state.w,
            angular_momentum_squared / (radius * radius * radius)
                - 1.0 / (radius * radius),
        };
    };
    auto relative_energy_error = [](const ev::CharacteristicState& state) {
        const double energy = 0.5 * state.w * state.w
            + 0.5 * angular_momentum_squared / (state.r * state.r)
            - 1.0 / state.r;
        constexpr double initial_energy =
            0.5 * angular_momentum_squared - 1.0;
        return std::abs((energy - initial_energy) / initial_energy);
    };

    const ev::RegularizedOrbitResult coarse =
        ev::advance_radial_characteristic_levi_civita_leapfrog(
            {1.0, 0.0}, half_period, rhs, 1, 128);
    const ev::RegularizedOrbitResult fine =
        ev::advance_radial_characteristic_levi_civita_leapfrog(
            {1.0, 0.0}, half_period, rhs, 256, 4096);
    assert(coarse.converged);
    assert(fine.converged);
    assert(fine.state.r > 0.0);
    assert(relative_energy_error(fine.state)
           < relative_energy_error(coarse.state));
    assert(relative_energy_error(fine.state) < 1.0e-3);
}

void test_adaptive_collision_shells() {
    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 1.0e6;
    p.bp_dsmc_shell_particles = 128;
    p.bp_dsmc_min_shell_particles = 16;

    std::vector<ev::SphericalDSMCParticle> particles(128);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        particles[i] = {
            0.1 + 0.02 * static_cast<double>(i),
            0.2 * std::sin(static_cast<double>(i)),
            0.05,
            static_cast<std::uint64_t>(i),
        };
    }
    const std::vector<ev::SphericalDSMCShell> shells =
        ev::spherical_dsmc_build_shells(particles, 1.0e-3, p);
    assert(shells.size() > 1);
    std::size_t cursor = 0;
    for (const ev::SphericalDSMCShell& shell : shells) {
        assert(shell.begin == cursor);
        assert(shell.end > shell.begin);
        assert(shell.end - shell.begin >= 16);
        assert(shell.density > 0.0);
        assert(shell.majorant_u_sigma > 0.0);
        cursor = shell.end;
    }
    assert(cursor == particles.size());
    bool has_resolution_limited_shell = false;
    for (const ev::SphericalDSMCShell& shell : shells) {
        const double width = shell.outer_edge - shell.inner_edge;
        has_resolution_limited_shell =
            has_resolution_limited_shell || width > shell.mean_free_path;
    }
    assert(has_resolution_limited_shell);
}

void test_velocity_averaged_mean_free_path_recovers_frank_and_evolves() {
    ::unsetenv("EV_SPHERICAL_DSMC_PAPER_EXACT");
    constexpr std::size_t count = 32;
    constexpr double particle_mass = 1.0e-3;
    std::vector<ev::SphericalDSMCParticle> slow(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double radius = 1.0 + 0.01 * static_cast<double>(i);
        slow[i] = {
            radius,
            i % 2 == 0 ? -0.2 : 0.2,
            0.1 * radius,
            static_cast<std::uint64_t>(i),
        };
    }

    ev::Params constant;
    constant.bp_cross_section = "constant";
    constant.bp_collision_strength = 1.0;
    constant.bp_sigma0 = 2.0;
    const ev::SphericalDSMCShell frank =
        ev::spherical_dsmc_describe_shell(
            slow, particle_mass, 0, slow.size(), constant);
    const double volume = (4.0 * ev::PI / 3.0)
        * std::pow(slow.back().r, 3);
    const double partner_number_density =
        static_cast<double>(count - 1) / volume;
    const double physical_pair_area =
        particle_mass * constant.bp_sigma0;
    const double frank_mean_free_path =
        1.0 / (partner_number_density * physical_pair_area);
    assert(nearly_equal(
        frank.mean_free_path, frank_mean_free_path, 2.0e-14));
    assert(frank.mean_free_path_pair_samples == count * (count - 1) / 2);
    assert(nearly_equal(
        frank.orbital_velocity_dispersion,
        std::sqrt((0.2 * 0.2 + 0.1 * 0.1) / 3.0),
        2.0e-14));

    ev::Params velocity_dependent = constant;
    velocity_dependent.bp_cross_section = "powerlaw";
    velocity_dependent.bp_sigma_u0 = 0.05;
    velocity_dependent.bp_sigma_n = 4.0;
    const ev::SphericalDSMCShell slow_shell =
        ev::spherical_dsmc_describe_shell(
            slow,
            particle_mass,
            0,
            slow.size(),
            velocity_dependent);
    std::vector<ev::SphericalDSMCParticle> fast = slow;
    for (auto& particle : fast) {
        particle.vr *= 4.0;
        particle.L *= 4.0;
    }
    const ev::SphericalDSMCShell fast_shell =
        ev::spherical_dsmc_describe_shell(
            fast,
            particle_mass,
            0,
            fast.size(),
            velocity_dependent);
    assert(nearly_equal(
        fast_shell.orbital_velocity_dispersion,
        4.0 * slow_shell.orbital_velocity_dispersion,
        2.0e-14));
    assert(fast_shell.mean_free_path > slow_shell.mean_free_path);
}

void test_paper_exact_unequal_mass_mfp_uses_pair_average() {
    EnvironmentRestorer restore("EV_SPHERICAL_DSMC_PAPER_EXACT");
    ::setenv("EV_SPHERICAL_DSMC_PAPER_EXACT", "1", 1);
    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 2.0;
    std::vector<ev::SphericalDSMCParticle> particles = {
        {1.0, -0.2, 0.1, 0, 1.0},
        {2.0, 0.2, 0.2, 1, 3.0},
    };
    const ev::SphericalDSMCShell shell =
        ev::spherical_dsmc_describe_shell(
            particles, 1.0, 0, particles.size(), p);
    const double volume =
        (4.0 * ev::PI / 3.0) * (8.0 - 1.0);
    const double pair_area =
        0.5 * (particles[0].mass + particles[1].mass) * p.bp_sigma0;
    const double partner_number_density = 1.0 / volume;
    assert(nearly_equal(
        shell.mean_free_path,
        1.0 / (partner_number_density * pair_area),
        2.0e-14));
    assert(shell.mean_free_path_pair_samples == 1);
    assert(shell.orbital_velocity_dispersion > 0.0);
}

void test_short_final_shell_is_merged() {
    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_collision_strength = 0.0;
    p.bp_sigma0 = 1.0;
    p.bp_dsmc_shell_particles = 64;
    p.bp_dsmc_min_shell_particles = 16;
    std::vector<ev::SphericalDSMCParticle> particles(130);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        particles[i] = {
            0.1 + 0.01 * static_cast<double>(i),
            0.0,
            0.1,
            static_cast<std::uint64_t>(i),
        };
    }
    const std::vector<ev::SphericalDSMCShell> shells =
        ev::spherical_dsmc_build_shells(particles, 1.0e-3, p);
    assert(shells.size() == 2);
    assert(shells[0].end - shells[0].begin == 64);
    assert(shells[1].end - shells[1].begin == 66);
}

void test_paper_exact_shell_extrema_and_four_vmax_majorant() {
    EnvironmentRestorer restore("EV_SPHERICAL_DSMC_PAPER_EXACT");
    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 0.5;
    p.bp_dsmc_shell_particles = 64;
    p.bp_dsmc_min_shell_particles = 16;
    std::vector<ev::SphericalDSMCParticle> particles(128);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const double radius = 1.0 + 0.01 * static_cast<double>(i);
        particles[i] = {
            radius,
            0.1 * std::sin(0.2 * static_cast<double>(i)),
            0.2 * radius,
            static_cast<std::uint64_t>(i),
        };
    }

    ::unsetenv("EV_SPHERICAL_DSMC_PAPER_EXACT");
    const std::vector<ev::SphericalDSMCShell> default_shells =
        ev::spherical_dsmc_build_shells(particles, 1.0e-6, p);
    ::setenv("EV_SPHERICAL_DSMC_PAPER_EXACT", "1", 1);
    const std::vector<ev::SphericalDSMCShell> paper_shells =
        ev::spherical_dsmc_build_shells(particles, 1.0e-6, p);

    assert(default_shells.size() == 2);
    assert(paper_shells.size() == 2);
    assert(default_shells[0].inner_edge == 0.0);
    assert(default_shells[0].outer_edge
           == 0.5 * (particles[63].r + particles[64].r));
    assert(paper_shells[0].inner_edge == particles[0].r);
    assert(paper_shells[0].outer_edge == particles[63].r);
    const double paper_volume = (4.0 * ev::PI / 3.0)
        * (std::pow(particles[63].r, 3)
           - std::pow(particles[0].r, 3));
    assert(nearly_equal(
        paper_shells[0].density,
        64.0e-6 / paper_volume,
        2.0e-15));
    assert(nearly_equal(
        paper_shells[0].majorant_u_sigma,
        2.0 * default_shells[0].majorant_u_sigma,
        2.0e-15));
}

void test_paper_exact_shell_cursor_resets_to_full_target() {
    EnvironmentRestorer restore("EV_SPHERICAL_DSMC_PAPER_EXACT");
    ::setenv("EV_SPHERICAL_DSMC_PAPER_EXACT", "1", 1);
    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 1.5e5;
    p.bp_dsmc_shell_particles = 4096;
    p.bp_dsmc_min_shell_particles = 32;

    // The first 4096-particle candidate is deliberately nonlocal, while its
    // first 2048-particle half is local.  From the accepted cursor at 2048,
    // the next full 4096-particle candidate is local.  Frank's moving-cursor
    // algorithm must therefore produce [0,2048) followed by [2048,6144).
    // A recursive fixed-parent tree incorrectly makes the second shell stop
    // at or before the fixed parent boundary 4096.
    std::vector<ev::SphericalDSMCParticle> particles(8192);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        double radius = 0.0;
        if (i < 2048) {
            radius = 1.0 + 9.0 * static_cast<double>(i) / 2047.0;
        } else if (i < 4096) {
            radius = 10.0
                + 0.01 * static_cast<double>(i - 2048) / 2047.0;
        } else if (i < 6144) {
            radius = 10.02
                + 89.98 * static_cast<double>(i - 4096) / 2047.0;
        } else {
            radius = 100.01
                + 99.99 * static_cast<double>(i - 6144) / 2047.0;
        }
        particles[i] = {
            radius,
            0.0,
            0.0,
            static_cast<std::uint64_t>(i),
        };
    }

    const std::vector<ev::SphericalDSMCShell> shells =
        ev::spherical_dsmc_build_shells(particles, 1.0e-6, p);
    assert(shells.size() >= 2);
    assert(shells[0].begin == 0);
    assert(shells[0].end == 2048);
    assert(shells[1].begin == 2048);
    assert(shells[1].end == 6144);
}

void test_paper_exact_cached_shell_descriptions_match_literal_rescans() {
    EnvironmentRestorer restore("EV_SPHERICAL_DSMC_PAPER_EXACT");
    ::setenv("EV_SPHERICAL_DSMC_PAPER_EXACT", "1", 1);
    ev::Params p;
    p.threads = 7;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 8.0e4;
    p.bp_dsmc_shell_particles = 4096;
    p.bp_dsmc_min_shell_particles = 32;
    std::vector<ev::SphericalDSMCParticle> particles(12345);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const double q = static_cast<double>(i);
        const double radius = 0.2 + 2.0e-4 * q + 3.0e-9 * q * q;
        particles[i] = {
            radius,
            0.3 * std::sin(0.017 * q),
            radius * (0.2 + 0.1 * std::cos(0.011 * q)),
            static_cast<std::uint64_t>(i),
        };
    }

    ev::SphericalDSMCShellWorkspace workspace;
    const std::vector<ev::SphericalDSMCShell> cached =
        ev::spherical_dsmc_build_shells(
            particles, 1.0e-6, p, workspace);
    std::vector<ev::SphericalDSMCShell> literal;
    const std::size_t target = 4096;
    const std::size_t minimum = 32;
    for (std::size_t begin = 0; begin < particles.size();) {
        std::size_t end = std::min(particles.size(), begin + target);
        ev::SphericalDSMCShell shell = ev::spherical_dsmc_describe_shell(
            particles, 1.0e-6, begin, end, p);
        while (shell.outer_edge - shell.inner_edge > shell.mean_free_path
               && end - begin >= 2 * minimum) {
            end = begin + (end - begin) / 2;
            shell = ev::spherical_dsmc_describe_shell(
                particles, 1.0e-6, begin, end, p);
        }
        literal.push_back(shell);
        begin = end;
    }

    assert(cached.size() == literal.size());
    for (std::size_t i = 0; i < cached.size(); ++i) {
        assert(cached[i].begin == literal[i].begin);
        assert(cached[i].end == literal[i].end);
        assert(cached[i].inner_edge == literal[i].inner_edge);
        assert(cached[i].outer_edge == literal[i].outer_edge);
        assert(cached[i].density == literal[i].density);
        assert(cached[i].mean_free_path == literal[i].mean_free_path);
        assert(cached[i].maximum_speed == literal[i].maximum_speed);
        assert(cached[i].maximum_speed_index
               == literal[i].maximum_speed_index);
        assert(cached[i].majorant_u_sigma == literal[i].majorant_u_sigma);
    }
}

void test_paper_exact_integer_remainder_trial_count() {
    std::vector<ev::SphericalDSMCParticle> particles(4);
    const std::uint64_t ids[4] = {2, 0, 3, 1};
    for (std::size_t i = 0; i < particles.size(); ++i) {
        particles[i].r = 1.0 + static_cast<double>(i);
        particles[i].id = ids[i];
    }
    ev::SphericalDSMCShell shell;
    shell.begin = 0;
    shell.end = particles.size();
    shell.density = 1.0;
    shell.majorant_u_sigma = 1.0;
    std::vector<double> remainder(particles.size(), 0.0);
    std::uint64_t total_trials = 0;
    for (int step = 0; step < 5; ++step) {
        total_trials += ev::spherical_dsmc_paper_trial_count(
            particles,
            shell,
            1.0 / 6.0,
            remainder);
    }
    // Each call contributes 0.25 trial.  Frank's strict A_shell>1 test
    // emits the carried trial on the fifth call, leaving 0.25 in aggregate.
    assert(total_trials == 1);
    for (double value : remainder) {
        assert(nearly_equal(value, 0.0625, 2.0e-15));
    }
}

void test_paper_exact_timestep_uses_vmax_not_four_vmax() {
    EnvironmentRestorer restore("EV_SPHERICAL_DSMC_PAPER_EXACT");
    ev::Params p;
    p.bp_dsmc_collision_epsilon = 1.0;
    ev::SphericalDSMCShell shell;
    shell.begin = 0;
    shell.end = 4;
    shell.density = 3.0;
    shell.majorant_u_sigma = 20.0;
    const double partner_number_density = 3.0 * 3.0 / 4.0;
    ::unsetenv("EV_SPHERICAL_DSMC_PAPER_EXACT");
    const double default_dt =
        ev::spherical_dsmc_collision_timestep({shell}, p);
    ::setenv("EV_SPHERICAL_DSMC_PAPER_EXACT", "1", 1);
    const double paper_dt =
        ev::spherical_dsmc_collision_timestep({shell}, p);
    assert(nearly_equal(
        default_dt, 1.0 / (partner_number_density * 20.0)));
    assert(nearly_equal(
        paper_dt, 4.0 / (partner_number_density * 20.0)));
}

void test_paper_exact_collision_invariants_and_deterministic_trials() {
    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 0.7;
    std::vector<ev::SphericalDSMCParticle> particles(32);
    double maximum_speed = 0.0;
    double radial_momentum_before = 0.0;
    double kinetic_before = 0.0;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const double q = static_cast<double>(i);
        const double radius = 0.8 + 0.01 * q;
        particles[i] = {
            radius,
            0.6 * std::sin(0.37 * q),
            radius * (0.3 + 0.2 * std::cos(0.19 * q)),
            static_cast<std::uint64_t>(i),
        };
        maximum_speed = std::max(
            maximum_speed,
            std::hypot(
                particles[i].vr,
                ev::spherical_dsmc_tangential_speed(particles[i])));
        radial_momentum_before += particles[i].vr;
        kinetic_before +=
            ev::spherical_dsmc_specific_kinetic_energy(particles[i]);
    }
    ev::SphericalDSMCShell shell;
    shell.begin = 0;
    shell.end = particles.size();
    shell.density = 2.0;
    shell.maximum_speed = maximum_speed;
    shell.majorant_u_sigma =
        ev::dsmc_u_sigma_majorant(4.0 * maximum_speed, p);
    const double dt = 30.0
        / (0.5 * 31.0 * shell.density * shell.majorant_u_sigma);
    std::vector<double> remainder(particles.size(), 0.0);
    const ev::DSMCCollisionStats stats =
        ev::spherical_dsmc_apply_shell_collisions_paper_exact(
            particles,
            shell,
            dt,
            p,
            UINT64_C(0x123456789abcdef0),
            remainder);
    assert(stats.trial_pairs == 30);
    assert(stats.accepted_pairs > 0);

    double radial_momentum_after = 0.0;
    double kinetic_after = 0.0;
    for (const ev::SphericalDSMCParticle& particle : particles) {
        radial_momentum_after += particle.vr;
        kinetic_after +=
            ev::spherical_dsmc_specific_kinetic_energy(particle);
    }
    assert(nearly_equal(
        radial_momentum_after,
        radial_momentum_before,
        2.0e-14));
    assert(nearly_equal(kinetic_after, kinetic_before, 2.0e-14));
    assert(stats.max_pair_momentum_error < 1.0e-12);
    assert(stats.max_pair_kinetic_energy_error < 1.0e-12);
}

void test_production_collision_path_accepts_unequal_masses() {
    ::unsetenv("EV_SPHERICAL_DSMC_PAPER_EXACT");
    ev::Params p;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 2.0;
    p.threads = 4;
    constexpr std::size_t count = 512;
    constexpr double fallback_mass = 1.0;
    std::vector<ev::SphericalDSMCParticle> particles(count);
    double momentum_before = 0.0;
    double kinetic_before = 0.0;
    std::vector<double> mass_before(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double phase = static_cast<double>(i);
        particles[i] = {
            0.9 + 0.2 * (phase + 0.5) / static_cast<double>(count),
            0.7 * std::sin(0.17 * phase),
            0.3 + 0.1 * std::cos(0.11 * phase),
            static_cast<std::uint64_t>(i),
            0.25 + 0.25 * static_cast<double>(i % 8),
        };
        const double mass = particles[i].mass;
        mass_before[i] = mass;
        momentum_before += mass * particles[i].vr;
        kinetic_before +=
            mass * ev::spherical_dsmc_specific_kinetic_energy(particles[i]);
    }
    const ev::SphericalDSMCShell shell = ev::spherical_dsmc_describe_shell(
        particles, fallback_mass, 0, particles.size(), p);
    const auto diagnostics = ev::spherical_dsmc_apply_collisions(
        particles,
        {shell},
        fallback_mass,
        0.002,
        7,
        p);
    assert(diagnostics.collision_accepted > 10);
    double momentum_after = 0.0;
    double kinetic_after = 0.0;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        assert(particles[i].mass == mass_before[i]);
        momentum_after += particles[i].mass * particles[i].vr;
        kinetic_after += particles[i].mass
            * ev::spherical_dsmc_specific_kinetic_energy(particles[i]);
    }
    assert(nearly_equal(momentum_after, momentum_before, 2.0e-13));
    assert(nearly_equal(kinetic_after, kinetic_before, 2.0e-13));
    assert(std::abs(diagnostics.collision_radial_momentum_error) < 2.0e-12);
    assert(std::abs(diagnostics.collision_kinetic_error) < 2.0e-12);
    assert(diagnostics.max_pair_momentum_error < 2.0e-12);
    assert(diagnostics.max_pair_kinetic_error < 2.0e-12);
}

void test_parallel_stable_sort_matches_serial_order() {
    ev::Params p;
    p.threads = 8;
    std::vector<ev::SphericalDSMCParticle> serial(4096);
    for (std::size_t i = 0; i < serial.size(); ++i) {
        serial[i] = {
            1.0 + 0.25 * std::sin(0.37 * static_cast<double>(i % 257)),
            0.01 * static_cast<double>(i),
            0.1,
            static_cast<std::uint64_t>(i),
        };
    }
    std::vector<ev::SphericalDSMCParticle> parallel = serial;
    ev::spherical_dsmc_sort_particles(serial, true);
    ev::SphericalDSMCSortWorkspace workspace;
    ev::spherical_dsmc_sort_particles_parallel(
        parallel,
        p,
        workspace,
        true);
    assert(parallel.size() == serial.size());
    for (std::size_t i = 0; i < serial.size(); ++i) {
        assert(parallel[i].r == serial[i].r);
        assert(parallel[i].vr == serial[i].vr);
        assert(parallel[i].L == serial[i].L);
        assert(parallel[i].id == serial[i].id);
    }
}

void test_paper_exact_parallel_path_is_literal_insertion_sort() {
    EnvironmentRestorer restore("EV_SPHERICAL_DSMC_PAPER_EXACT");
    ::setenv("EV_SPHERICAL_DSMC_PAPER_EXACT", "1", 1);
    ev::Params p;
    p.threads = 8;
    std::vector<ev::SphericalDSMCParticle> ordered(4096);
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        ordered[i] = {
            // Repeated radii exercise stable equal-key ordering as well as
            // the deterministic merges between insertion-sorted lanes.
            1.0 + 1.0e-4 * static_cast<double>(i / 3),
            0.01 * static_cast<double>(i),
            0.1,
            static_cast<std::uint64_t>(i),
        };
    }
    std::vector<ev::SphericalDSMCParticle> expected;
    expected.reserve(ordered.size());
    for (std::size_t i = 0; i < ordered.size() / 2; ++i) {
        expected.push_back(ordered[ordered.size() / 2 + i]);
        expected.push_back(ordered[i]);
    }
    std::vector<ev::SphericalDSMCParticle> actual = expected;
    ev::spherical_dsmc_insertion_sort_particles(expected);
    ev::SphericalDSMCSortWorkspace workspace;
    ev::spherical_dsmc_sort_particles_parallel(actual, p, workspace, false);
    assert(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        assert(actual[i].r == expected[i].r);
        assert(actual[i].vr == expected[i].vr);
        assert(actual[i].L == expected[i].L);
        assert(actual[i].id == expected[i].id);
    }
}

void test_cached_collision_timestep_matches_fresh_shell_rebuild() {
    ev::Params p;
    p.threads = 4;
    p.bp_cross_section = "constant";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 0.5;
    p.bp_dsmc_collision_epsilon = 1.0;
    p.bp_dsmc_shell_particles = 64;
    p.bp_dsmc_min_shell_particles = 16;
    p.bp_dsmc_seed = 12345;
    std::vector<ev::SphericalDSMCParticle> particles(256);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const double q = static_cast<double>(i);
        particles[i] = {
            0.5 + 0.01 * q,
            0.4 * std::sin(0.23 * q),
            0.05 + 0.02 * std::cos(0.17 * q),
            static_cast<std::uint64_t>(i),
        };
    }
    constexpr double particle_mass = 1.0e-3;
    ev::SphericalDSMCShellWorkspace shell_workspace;
    std::vector<ev::SphericalDSMCShell> shells =
        ev::spherical_dsmc_build_shells(
            particles,
            particle_mass,
            p,
            shell_workspace);
    ev::SphericalDSMCCollisionWorkspace collision_workspace;
    ev::spherical_dsmc_apply_collisions(
        particles,
        shells,
        particle_mass,
        0.5,
        7,
        p,
        collision_workspace);
    const double cached =
        ev::spherical_dsmc_collision_timestep_after_collisions(
            shells,
            collision_workspace,
            p);
    ev::SphericalDSMCShellWorkspace rebuilt_workspace;
    const std::vector<ev::SphericalDSMCShell> rebuilt =
        ev::spherical_dsmc_build_shells(
            particles,
            particle_mass,
            p,
            rebuilt_workspace);
    const double direct = ev::spherical_dsmc_collision_timestep(rebuilt, p);
    assert(cached == direct);
}

void test_frank_central_density_uses_innermost_250() {
    std::vector<ev::SphericalDSMCParticle> particles(5000);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        particles[i].r = static_cast<double>(i + 1);
        particles[i].id = static_cast<std::uint64_t>(i);
    }
    constexpr double particle_mass = 2.0;
    const double expected = 250.0 * particle_mass
        / ((4.0 * ev::PI / 3.0) * std::pow(250.0, 3));
    const double measured =
        ev::spherical_dsmc_central_density(particles, particle_mass);
    assert(nearly_equal(measured, expected, 1.0e-15));
}

void test_restart_round_trip_and_replacement() {
    ev::Params p;
    p.central_mass = 0.4;
    const double particle_mass = 2.5e-4;
    std::vector<ev::SphericalDSMCParticle> particles =
        circular_markers(32, particle_mass, p);
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "venus_spherical_dsmc_restart_test";
    const std::filesystem::path path = directory / "restart.bin";
    std::filesystem::remove_all(directory);

    ev::spherical_dsmc_write_restart(
        path, particles, particle_mass, 1.25, 17, -0.75, 1234, 56, 7);
    ev::spherical_dsmc_write_restart(
        path, particles, particle_mass, 1.5, 18, -0.75, 1240, 57, 8);
    std::vector<ev::SphericalDSMCParticle> restored;
    const ev::SphericalDSMCRestartHeader header =
        ev::spherical_dsmc_read_restart(path, restored);

    assert(header.particle_count == particles.size());
    assert(nearly_equal(header.particle_mass, particle_mass));
    assert(nearly_equal(header.time, 1.5));
    assert(header.step == 18);
    assert(header.collision_accepted_total == 1240);
    assert(header.lc_retries_total == 57);
    assert(header.lc_unresolved_total == 8);
    assert(header.version == 4);
    assert(!header.next_collision_dt.has_value());
    assert(restored.size() == particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i) {
        assert(restored[i].id == particles[i].id);
        assert(restored[i].r == particles[i].r);
        assert(restored[i].vr == particles[i].vr);
        assert(restored[i].L == particles[i].L);
        assert(restored[i].mass == particle_mass);
    }


    const std::filesystem::path paper_path = directory / "restart_v5.bin";
    std::vector<double> remainder(particles.size());
    for (std::size_t i = 0; i < remainder.size(); ++i) {
        remainder[i] = 0.25 * static_cast<double>(i) / remainder.size();
    }
    ev::spherical_dsmc_write_restart(
        paper_path,
        particles,
        particle_mass,
        2.0,
        19,
        -0.75,
        1250,
        58,
        0,
        &remainder);
    std::vector<ev::SphericalDSMCParticle> paper_restored;
    std::vector<double> remainder_restored;
    const ev::SphericalDSMCRestartHeader paper_header =
        ev::spherical_dsmc_read_restart(
            paper_path,
            paper_restored,
            &remainder_restored);
    assert(paper_header.version == 5);
    assert(remainder_restored == remainder);
    assert(!paper_header.next_collision_dt.has_value());

    const auto clock_path = directory / "restart_clock.bin";
    ev::spherical_dsmc_write_restart(clock_path, particles, particle_mass,
        2.0, 19, -0.75, 1250, 58, 0, nullptr, 0.125, UINT64_C(1234));
    const auto clock_header = ev::spherical_dsmc_read_restart(clock_path, restored);
    assert(clock_header.version == 6);
    assert(clock_header.next_collision_dt == 0.125);
    assert(clock_header.collision_control_fingerprint == UINT64_C(1234));
    ev::spherical_dsmc_write_restart(clock_path, particles, particle_mass,
        2.0, 19, -0.75, 1250, 58, 0, &remainder, 0.25, UINT64_C(4321));
    const auto paper_clock_header = ev::spherical_dsmc_read_restart(
        clock_path, restored, &remainder_restored);
    assert(paper_clock_header.version == 7);
    assert(paper_clock_header.next_collision_dt == 0.25);
    assert(paper_clock_header.collision_control_fingerprint == UINT64_C(4321));
    assert(remainder_restored == remainder);

    auto rewrite_valid = [&]() {
        ev::spherical_dsmc_write_restart(
            path,
            particles,
            particle_mass,
            1.5,
            18,
            -0.75,
            1240,
            57,
            8);
    };
    auto overwrite = [&](std::streamoff offset, const auto& value) {
        std::fstream stream(
            path,
            std::ios::binary | std::ios::in | std::ios::out);
        assert(stream.good());
        stream.seekp(offset);
        stream.write(
            reinterpret_cast<const char*>(&value),
            static_cast<std::streamsize>(sizeof(value)));
        stream.close();
        assert(stream.good());
    };
    auto rejected_restart = [&]() {
        std::vector<ev::SphericalDSMCParticle> ignored;
        try {
            (void)ev::spherical_dsmc_read_restart(path, ignored);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };

    // A full-width field with the expected prefix but no NUL terminator must be
    // rejected without invoking an unbounded C-string read.
    rewrite_valid();
    const std::array<char, 16> unterminated_magic{
        'V', 'E', 'N', 'U', 'S', 'S', 'P', 'H',
        'D', 'S', 'M', 'C', '1', 'X', 'X', 'X'};
    overwrite(0, unterminated_magic);
    assert(rejected_restart());

    // Header counts are bounded against the actual record payload before a
    // vector allocation, and the persisted clock must narrow safely.
    rewrite_valid();
    const std::uint64_t impossible_count =
        std::numeric_limits<std::uint64_t>::max();
    overwrite(20, impossible_count);
    assert(rejected_restart());

    rewrite_valid();
    const std::uint64_t truncated_count = 2;
    overwrite(20, truncated_count);
    assert(rejected_restart());

    rewrite_valid();
    const double invalid_time =
        std::numeric_limits<double>::quiet_NaN();
    overwrite(36, invalid_time);
    assert(rejected_restart());

    rewrite_valid();
    const std::int64_t invalid_step =
        std::numeric_limits<std::int64_t>::max();
    overwrite(44, invalid_step);
    assert(rejected_restart());

    // An inactive collision operator legitimately has no finite collision
    // limit. Nonpositive and NaN persisted limits are corrupt clock states.
    ev::spherical_dsmc_write_restart(path, particles, particle_mass,
        1.5, 18, -0.75, 1240, 57, 8, nullptr,
        std::numeric_limits<double>::infinity(), UINT64_C(1234));
    assert(std::isinf(ev::spherical_dsmc_read_restart(path, restored)
                         .next_collision_dt.value()));
    for (double invalid_clock : {0.0, -1.0,
             std::numeric_limits<double>::quiet_NaN()}) {
        ev::spherical_dsmc_write_restart(path, particles, particle_mass,
            1.5, 18, -0.75, 1240, 57, 8, nullptr, 0.125, UINT64_C(1234));
        overwrite(84, invalid_clock);
        assert(rejected_restart());
    }

    // Original 2/3 records have one header mass and no per-marker mass.
    // Build those exact layouts from the fixed-width 4 header/records.
    rewrite_valid();
    std::ifstream legacy_source(path, std::ios::binary);
    const std::vector<char> old_layout{
        std::istreambuf_iterator<char>(legacy_source),
        std::istreambuf_iterator<char>()};
    for (std::uint32_t version : {2U, 3U}) {
        auto prefix = std::vector<char>(old_layout.begin(), old_layout.begin() + 84);
        std::memcpy(prefix.data() + 16, &version, sizeof(version));
        const auto legacy_path = directory / ("legacy_" + std::to_string(version));
        std::ofstream legacy(legacy_path, std::ios::binary);
        legacy.write(prefix.data(), prefix.size());
        for (std::size_t i = 0; i < particles.size(); ++i) {
            legacy.write(old_layout.data() + 84 + 40 * i, 32);
        }
        if (version == 3) {
            ev::write_binary(legacy, static_cast<std::uint64_t>(remainder.size()));
            for (double value : remainder) ev::write_binary(legacy, value);
        }
        legacy.close();
        const auto legacy_header = ev::spherical_dsmc_read_restart(
            legacy_path, restored, &remainder_restored);
        assert(legacy_header.version == version);
        assert(!legacy_header.next_collision_dt.has_value());
        assert(restored.size() == particles.size());
        for (std::size_t i = 0; i < particles.size(); ++i) {
            assert(restored[i].mass == particle_mass);
            assert(restored[i].r == particles[i].r);
        }
        if (version == 3) assert(remainder_restored == remainder);
    }
    std::filesystem::remove_all(directory);
}

void test_restart_preserves_collision_clock_in_actual_solver(bool paper_exact = true) {
    EnvironmentRestorer restore_paper("EV_SPHERICAL_DSMC_PAPER_EXACT");
    if (paper_exact) ::setenv("EV_SPHERICAL_DSMC_PAPER_EXACT", "1", 1);
    else ::unsetenv("EV_SPHERICAL_DSMC_PAPER_EXACT");
    ev::Params p;
    p.physics_model = "boltzmann_poisson";
    p.bp_collision_operator = "spherical_dsmc";
    p.vp_universe = "static";
    p.vp_poisson_source = "total_density";
    p.threads = 1;
    p.vp_gravity = 0.001;
    p.central_mass = 0.0;
    p.vp_orbit_regularization = "auto";
    p.vp_orbit_energy_tolerance = 0.01;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    p.bp_collision_strength = 1.0;
    p.bp_sigma0 = 1000.0;
    p.bp_dsmc_seed = 12345;
    p.dt_max = 1.0;
    p.t_end = 10.0;
    p.save_every = 1;
    p.restart_every = 1;
    p.Nr = 16;
    p.Nw = 16;
    p.NL = 4;
    p.r_min = 0.01;
    p.r_max = 4.0;
    p.w_max = 2.0;
    p.L_max = 1.0;
    constexpr double particle_mass = 0.001;
    std::vector<ev::SphericalDSMCParticle> particles(128);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const double q = static_cast<double>(i);
        const double radius = 0.8 + 0.002 * q;
        particles[i] = {radius, 0.6 * std::sin(0.37 * q),
            radius * (0.3 + 0.2 * std::cos(0.19 * q)),
            static_cast<std::uint64_t>(i)};
    }
    const auto directory = std::filesystem::temp_directory_path()
        / (paper_exact ? "venus_spherical_dsmc_clock_continuation_test_paper"
                       : "venus_spherical_dsmc_clock_continuation_test_generic");
    std::filesystem::remove_all(directory);
    const auto input = directory / "input/restart.bin";
    const auto initial_energy = ev::spherical_dsmc_total_energy(
        particles, particle_mass, p);
    std::vector<double> remainder(particles.size(), 0.0);
    // An independent initial ensemble starts with an empty collision interval.
    ev::spherical_dsmc_write_restart(input, particles, particle_mass, 0.0,
        0, initial_energy.total, 0, 0, 0, paper_exact ? &remainder : nullptr);
    auto run = [&](const std::string& name, const std::filesystem::path& source,
                   int end_step, double sigma = 1000.0) {
        ev::Params options = p;
        options.output_dir = (directory / name).string();
        options.restart_dir = (directory / name / "restarts").string();
        options.restart_from = source.string();
        options.max_steps = end_step;
        options.bp_sigma0 = sigma;
        auto state = ev::create_solver_state(options, ev::MpiContext{});
        ev::run_spherical_dsmc_solver(state);
        return state.restart_path;
    };
    const auto continuous = run("continuous", input, 6);
    const auto split = run("split", input, 3);
    const auto continued = run("continued", split, 6);
    std::vector<ev::SphericalDSMCParticle> reference, resumed;
    const auto before = ev::spherical_dsmc_read_restart(continuous, reference);
    const auto after = ev::spherical_dsmc_read_restart(continued, resumed);
    std::cerr << std::setprecision(17)
        << "SphDSMC split-clock paper_exact=" << paper_exact
        << " final times: continuous=" << before.time
        << " resumed=" << after.time << " accepted collisions="
        << before.collision_accepted_total << '/' << after.collision_accepted_total
        << '\n';
    // Stop at an accepted natural step, so neither branch shortens its step
    // to hit an imposed time. The only process boundary is the checkpoint.
    assert(before.time == after.time);
    assert(before.version == (paper_exact ? 7U : 6U));
    assert(before.step == after.step && before.step == 6);
    assert(before.collision_accepted_total > 0);
    assert(before.collision_accepted_total == after.collision_accepted_total);
    for (std::size_t i = 0; i < reference.size(); ++i) {
        assert(reference[i].id == resumed[i].id);
        assert(reference[i].r == resumed[i].r);
        assert(reference[i].vr == resumed[i].vr);
        assert(reference[i].L == resumed[i].L);
        assert(reference[i].mass == resumed[i].mass);
    }
    auto bytes = [](const std::filesystem::path& path) {
        std::ifstream stream(path, std::ios::binary);
        return std::vector<char>{std::istreambuf_iterator<char>(stream),
                                 std::istreambuf_iterator<char>()};
    };
    assert(bytes(continuous) == bytes(continued));
    bool rejected_changed_law = false;
    const auto accepted_bytes = bytes(split);
    try {
        (void)run("changed_cross_section", split, 6, 500.0);
    } catch (const std::runtime_error& error) {
        rejected_changed_law = std::string(error.what()).find(
            "restart collision controls differ") != std::string::npos;
    }
    assert(rejected_changed_law);
    assert(bytes(split) == accepted_bytes);
    std::filesystem::remove_all(directory);
}

void test_spherical_dsmc_rejects_unimplemented_gravity_semantics() {
    ev::Params p;
    p.physics_model = "boltzmann_poisson";
    p.bp_collision_operator = "spherical_dsmc";
    p.threads = 1;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";

    p.vp_universe = "expanding";
    bool rejected_expanding = false;
    try {
        ev::validate_params(p);
    } catch (const std::runtime_error&) {
        rejected_expanding = true;
    }
    assert(rejected_expanding);

    p.vp_universe = "static";
    p.vp_poisson_source = "density_contrast";
    bool rejected_contrast = false;
    try {
        ev::validate_params(p);
    } catch (const std::runtime_error&) {
        rejected_contrast = true;
    }
    assert(rejected_contrast);

    p.vp_poisson_source = "total_density";
    ev::validate_params(p);
}

void test_spherical_dsmc_rejects_inconsistent_angular_kernel() {
    ev::Params p;
    p.physics_model = "boltzmann_poisson";
    p.bp_collision_operator = "spherical_dsmc";
    p.threads = 1;
    p.vp_universe = "static";
    p.vp_poisson_source = "total_density";

    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "yukawa_born";
    bool rejected_constant_yukawa = false;
    try {
        ev::validate_params(p);
    } catch (const std::runtime_error&) {
        rejected_constant_yukawa = true;
    }
    assert(rejected_constant_yukawa);

    p.bp_cross_section = "yukawa_born";
    p.bp_boltzmann_kernel = "isotropic";
    bool rejected_yukawa_isotropic = false;
    try {
        ev::validate_params(p);
    } catch (const std::runtime_error&) {
        rejected_yukawa_isotropic = true;
    }
    assert(rejected_yukawa_isotropic);
}

void test_spherical_dsmc_metadata_reports_effective_marker_backend() {
    ev::Params p;
    p.physics_model = "boltzmann_poisson";
    p.bp_collision_operator = "spherical_dsmc";
    p.vp_universe = "static";
    p.vp_poisson_source = "total_density";
    p.inner_boundary = "delayed_return";
    p.outer_boundary = "reflecting";
    ev::MpiContext mpi;
    const ev::LDecomposition decomp =
        ev::make_l_decomposition(p.NL, mpi);
    const std::filesystem::path path =
        std::filesystem::temp_directory_path()
        / "venus_spherical_dsmc_metadata_test.json";
    ev::write_metadata(
        path,
        p,
        p,
        decomp,
        mpi,
        17,
        3.5,
        0.125,
        5,
        4,
        true,
        true,
        true);
    std::ifstream input(path);
    const std::string text{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    assert(text.find(
        "\"time_integrator\": "
        "\"spherical_dsmc_self_gravitating_kick_drift_kick_local_ntc\"")
        != std::string::npos);
    assert(text.find("\"bp_dsmc_resolved_marker_count\": 17")
           != std::string::npos);
    assert(text.find("\"bp_dsmc_resolved_total_mass\": 3.5")
           != std::string::npos);
    assert(text.find("\"bp_dsmc_initial_common_marker_mass\": 0.125")
           != std::string::npos);
    assert(text.find("\"bp_dsmc_restart_output_version\": 5")
           != std::string::npos);
    assert(text.find("\"bp_dsmc_restart_input_version\": 4")
           != std::string::npos);
    assert(text.find("\"bp_dsmc_paper_exact\": true")
           != std::string::npos);
    assert(text.find("\"bp_dsmc_force_consistent_orbit_audit\": true")
           != std::string::npos);
    assert(text.find("\"bp_dsmc_force_radix_sort\": true")
           != std::string::npos);
    assert(text.find("\"inner_boundary\": \"origin_reflection_fixed\"")
           != std::string::npos);
    assert(text.find(
        "\"outer_boundary\": \"unbounded_marker_orbits_no_outer_wall\"")
        != std::string::npos);
    assert(text.find(
        "\"decomposition\": \"single_process_persistent_markers_openmp\"")
        != std::string::npos);
    std::filesystem::remove(path);
}

void test_origin_collisions_preserve_stored_kinetic_energy() {
    ev::Params p;
    p.bp_sigma0 = 2.0;
    p.bp_collision_strength = 1.0;
    p.bp_cross_section = "constant";
    p.bp_boltzmann_kernel = "isotropic";
    std::vector<ev::SphericalDSMCParticle> initial(128);
    for (std::size_t i = 0; i < initial.size(); ++i) {
        initial[i] = {0.0, 0.5 + 0.01 * i, 0.0, i, 0.01};
    }
    auto energy = [](const auto& particles) {
        long double result = 0.0L;
        for (const auto& particle : particles) {
            result += particle.mass * ev::spherical_dsmc_specific_kinetic_energy(particle);
        }
        return result;
    };
    const long double before = energy(initial);
    ev::SphericalDSMCShell shell;
    shell.begin = 0;
    shell.end = initial.size();
    shell.number_density = 1.0;
    shell.minimum_mass = shell.maximum_mass = 0.01;
    shell.maximum_speed = 1.77;
    shell.majorant_u_sigma = 0.08;
    for (bool paper_exact : {false, true}) {
        auto particles = initial;
        ev::DSMCCollisionStats diagnostics;
        if (paper_exact) {
            std::vector<double> remainder(particles.size(), 0.0);
            diagnostics = ev::spherical_dsmc_apply_shell_collisions_paper_exact(
                particles, shell, 0.01, 10.0, p, 10, remainder);
        } else {
            ev::SphericalDSMCCollisionWorkspace::ThreadVelocityCache cache;
            diagnostics = ev::spherical_dsmc_apply_shell_collisions_lazy(
                particles, shell, 0.01, 10.0, p, 9, 10, cache);
        }
        assert(diagnostics.accepted_pairs > 0);
        assert(std::abs(energy(particles) - before) < 2.0e-14L * before);
        for (const auto& particle : particles) {
            assert(std::isfinite(particle.vr) && particle.L == 0.0);
        }
    }
    ev::SphericalDSMCParticle invalid = initial[0];
    invalid.L = 1.0;
    bool rejected = false;
    try { (void)ev::spherical_dsmc_tangential_speed(invalid); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    // The radial Kepler kinetic term has a finite limit even when r^2
    // underflows. Zero enclosed mass leaves exactly radial kinetic energy.
    auto radial = initial[0];
    radial.r = 1.0e-200;
    p.vp_gravity = 1.0;
    assert(ev::spherical_dsmc_specific_kepler_energy(radial, 0.0, p)
        == 0.5 * radial.vr * radial.vr);
    for (auto& particle : initial) particle.vr = 1.0;
    p.bp_cross_section = "powerlaw";
    p.bp_sigma_u0 = 1.0;
    const auto mean_free_path = ev::spherical_dsmc_velocity_averaged_mean_free_path(
        initial, 0.01, 0, initial.size(), 1.0, p);
    assert(std::isfinite(mean_free_path.mean_free_path)
        && mean_free_path.mean_free_path > 0.0);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--restart-clock-only") {
        test_restart_preserves_collision_clock_in_actual_solver();
        test_restart_preserves_collision_clock_in_actual_solver(false);
        std::cout << "spherical DSMC collision-clock continuation test passed\n";
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--restart-only") {
        test_restart_round_trip_and_replacement();
        std::cout << "spherical DSMC restart tests passed\n";
        return 0;
    }
    test_origin_collisions_preserve_stored_kinetic_energy();
    test_circular_orbits_are_fixed_points();
    test_origin_endpoint_failure_restores_particles();
    test_frank_equation5_particle_potential();
    test_force_consistent_orbit_audit_removes_irreducible_half_self();
    test_paper_orbit_energy_defect_parabolic_guard();
    test_lc_retry_bookkeeping_is_retry_local();
    test_paper_lc_leapfrog_converges_through_pericentre();
    test_adaptive_collision_shells();
    test_velocity_averaged_mean_free_path_recovers_frank_and_evolves();
    test_paper_exact_unequal_mass_mfp_uses_pair_average();
    test_short_final_shell_is_merged();
    test_paper_exact_shell_extrema_and_four_vmax_majorant();
    test_paper_exact_shell_cursor_resets_to_full_target();
    test_paper_exact_cached_shell_descriptions_match_literal_rescans();
    test_paper_exact_integer_remainder_trial_count();
    test_paper_exact_timestep_uses_vmax_not_four_vmax();
    test_paper_exact_collision_invariants_and_deterministic_trials();
    test_production_collision_path_accepts_unequal_masses();
    test_parallel_stable_sort_matches_serial_order();
    test_paper_exact_parallel_path_is_literal_insertion_sort();
    test_cached_collision_timestep_matches_fresh_shell_rebuild();
    test_frank_central_density_uses_innermost_250();
    test_restart_round_trip_and_replacement();
    test_restart_preserves_collision_clock_in_actual_solver();
    test_restart_preserves_collision_clock_in_actual_solver(false);
    test_spherical_dsmc_rejects_unimplemented_gravity_semantics();
    test_spherical_dsmc_rejects_inconsistent_angular_kernel();
    test_spherical_dsmc_metadata_reports_effective_marker_backend();
    std::cout << "spherical DSMC solver tests passed\n";
    return 0;
}
